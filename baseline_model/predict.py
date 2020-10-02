from PIL import Image
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torchvision import datasets, models, transforms
import torch
import net_distill
from torch.autograd import Variable as V

def returnTF():
  tf = trn.Compose([
      trn.Resize((224,224)),
      trn.ToTensor(),
      trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  return tf


def predict(img_url, labels_url, student_model, model_A):
  tf = returnTF()
  img = Image.open(img_url)
  img = img.convert('RGB')
  input_img = V(tf(img).unsqueeze(0))

  with open(labels_url, "r") as f:
    classes=[label.split('\n')[0] for label in f]
  if student_model == '3conv':       
    model = net_distill.Net_3conv(num_channels = 16, num_classes = len(classes))
  if student_model == '5conv':       
    model = net_distill.Net_5conv(num_channels = 16, num_classes = len(classes))
  if student_model == '7conv':       
    model = net_distill.Net_7conv(num_channels = 16, num_classes = len(classes))
  model.eval()
  checkpoint = torch.load(model_A, map_location=torch.device('cpu'))
  model.load_state_dict(checkpoint)

  output = model.forward(input_img)
  prediction = int(torch.max(output.data, 1)[1].numpy())
  return classes[prediction]