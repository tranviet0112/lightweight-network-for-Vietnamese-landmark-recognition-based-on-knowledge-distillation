from PIL import Image
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torchvision import datasets, models, transforms
import res18_FNN
import res50_FNN
from extract_feature import extract_feature
import torch


def predict(img_url, model_, model_distil_CNN, labels_url, model_FNN):
      feature = extract_feature(img_url, model_, model_distil_CNN)  
      with open(labels_url, "r") as f:
            classes=[label.split('\n')[0] for label in f]
      if model_ == 'resnet18':
            model = res18_FNN.MyFNN(512, 256, 128, len(classes)) 
      if model_ == 'resnet50':
            model = res50_FNN.MyFNN(2048, 512, 256, 128, len(classes))
      model.eval()
      checkpoint = torch.load(model_FNN)
      model.load_state_dict(checkpoint)
      output = model(torch.tensor(feature).unsqueeze(0))
      prediction = int(torch.max(output.data, 1)[1].numpy())

      return classes[prediction]

