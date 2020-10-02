from PIL import Image
from torch.autograd import Variable as V
from torchvision import transforms as transforms
from torchvision import datasets, models, transforms
from pathlib import Path
import torch
import numpy as np
from torch.nn import functional as F
import wideresnet

def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def returnTF():
# load the image transformer
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return tf

def load_model(path, model):
    model_file = path
    if model == 'resnet18':
        model = wideresnet.resnet18(num_classes=365)
    if model == 'resnet50':
        model = wideresnet.resnet50(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    #model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(state_dict)
    for i, (name, module) in enumerate(model._modules.items()):
      module = recursion_change_bn(model)
    model.eval()
    return model

def extract_feature(img_url, model_, path_model_place365):
    tf = returnTF()
    model = load_model(path_model_place365, model_)
    
    class SaveOutput:
        def __init__(self):
            self.outputs = []
            
        def __call__(self, module, module_in, module_out):
            self.outputs.append(module_out)
            
        def clear(self):
            self.outputs = []

    save_output = SaveOutput()
    hook_handles = []
    for layer in model.modules():

        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)
  
    # Load images
    img = Image.open(img_url)
    
    # Handle some special image format in PIL Image library and convert them into RGB to process properly
    img = img.convert('RGB')
    input_img = V(tf(img).unsqueeze(0))
    logit = model.forward(input_img)
    if model_ == 'resnet18':
        raw_feature = save_output.outputs[-2].view(-1, 512)
    if model_ == 'resnet50':
        raw_feature = save_output.outputs[-2].view(-1, 2048)
    raw_feature = np.squeeze(raw_feature.detach().cpu().numpy())
    return raw_feature
    
