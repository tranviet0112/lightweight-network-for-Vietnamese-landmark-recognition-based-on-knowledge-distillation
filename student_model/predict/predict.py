from PIL import Image
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torchvision import datasets, models, transforms
import FNN
from extract_feature import extract_feature
import torch


def predict(img_url, model_distil_CNN, studend_model, num_classes_extract_step, num_channels, labels_url, model_FNN):
  feature = extract_feature(img_url, model_distil_CNN, studend_model, num_classes_extract_step, num_channels)  
  with open(labels_url, "r") as f:
        classes=[label.split('\n')[0] for label in f]
  model = FNN.MyFNN(512, 256, 128, len(classes))
  model.eval()
  checkpoint = torch.load(model_FNN)
  model.load_state_dict(checkpoint)
  output = model(torch.tensor(feature).unsqueeze(0))
  prediction = int(torch.max(output.data, 1)[1].numpy())
  return classes[prediction]

