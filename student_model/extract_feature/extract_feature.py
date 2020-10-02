# PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
# by Bolei Zhou, sep 2, 2017
# updated, making it compatible to pytorch 1.x in a hacky way

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
import cv2
from PIL import Image
from multiprocessing import Pool
from dataclasses import dataclass
import torch.tensor as tensor
import argparse
import net_distill

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=None, 
                    help="The directory containing the dataset needs to extract feature")
parser.add_argument('--model_dir', default='best.pth.tar',
                    help="path of weight")
parser.add_argument('--root_output', default=None,
                    help="directory for output")
parser.add_argument('--student_model', default=None,
                    help="model of student: 3CNN or 5CNN or 7CNN")
parser.add_argument('--num_classes', default=None,
                    help="the number of labels of student model")
parser.add_argument('--num_channels', default=None,
                    help="param for student model architecture, is set in KD model")
parser.add_argument('--subset', default=(0, 3),
                     help=" [important] b/c of the memory overflow problem, so in the dataset directory has many labels, \
                        we extract each subset start_end passed in tuple, \
                        for example we want to extract the first 3 labels --subset (0, 3).")  

def returnTF():
# load the image transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf

def load_model(model_file):
    model_file = model_file
    if args.student_model == '3CNN':
        model = net_distill.Net_3CNN(int(args.num_classes), int(args.num_channels))
    elif args.student_model == '5CNN':
        model = net_distill.Net_5CNN(int(args.num_classes), int(args.num_channels))
    elif args.student_model == '7CNN':
        model = net_distill.Net_7CNN(int(args.num_classes), int(args.num_channels))
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)    
    model.eval()
    return model
# load the transformer
tf = returnTF() # image transformer

def extract_placeCNN_feature(model, image_file_path, raw_feat_output_path):
    class SaveOutput:
        def __init__(self):
            self.outputs = []
            
        def __call__(self, module, module_in, module_out):
            self.outputs.append(module_out)
            
        def clear(self):
            self.outputs = []
    # chứa tất cả thông tin tất cả layer
    save_output = SaveOutput()
    hook_handles = []
    for layer in model.modules():
        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)

    # Load images
    img = Image.open(image_file_path)  
    # Handle some special image format in PIL Image library and convert them into RGB to process properly
    img = img.convert('RGB')
    input_img = V(tf(img).unsqueeze(0))
    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    # # sẽ append layer cần lấy vào đây
    features_blobs = []
    # các layer cần lấy, thiết kế là một list vì có lúc cần lấy nhiều layer
    features_names = [-2] # 
    # ở đây e chọn -2, vì theo kiến trúc mạng mình thiết kế thì 2 lớp cuối là avgpool2d và fc
    for name in features_names:
        features_blobs.append(save_output.outputs[name])    # save_output.outputs[name] là các tensor

    raw_feature = features_blobs[0].view(-1, 512) # vì đây là lớp avgpool2d, view để ra vector 512
    # chuyển tensor về numpy để save
    raw_feature = np.squeeze(raw_feature.detach().cpu().numpy())
    np.save(raw_feat_output_path, raw_feature)
    return 1


if __name__ == '__main__':

    # load the model
    args = parser.parse_args()
    file_model = args.model_dir
    model = load_model(file_model)
    import glob
    # load the test image
    root_output = args.root_output
    link_train_folder = args.data_dir
    train_folder = list(sorted(glob.glob(f'{link_train_folder}\\*')))
    index = eval(args.subset)
    def run(image_folder):
        class_folder = image_folder.split('\\')[-1]
        img_urls = list(sorted(glob.glob(f'{image_folder}/*')))
        count = 0
        for img_url in img_urls:
            file_name = img_url.split('\\')[-1].split('.')[0]
            raw_output_folder_path = os.path.join(root_output, 'Raw', class_folder)
            if not os.path.exists(raw_output_folder_path):
                os.makedirs(raw_output_folder_path)
            k = extract_placeCNN_feature(model, img_url, os.path.join(raw_output_folder_path, file_name + '.npy'))
            count += 1
            del k      
              
        print('Done {} image of {}'.format(count, class_folder))
    
    for image_folder in train_folder[index[0]: index[1]]:
        run(image_folder)
    print('Completed label {} to label {}'.format(index[0], index[1]-1))