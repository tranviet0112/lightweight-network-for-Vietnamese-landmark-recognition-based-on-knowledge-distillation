import torch
import torchvision
import torchvision.transforms as transforms
import glob
from predict import predict
import time

model_place365 = 'resnet50_places365.pth.tar'   #'wideresnet18_places365.pth.tar or resnet50_places365.pth.tar'
model_ =    'resnet50'        #'resnet18 or resnet50'
labels_url = 'labels_res50.txt'   #'labels file from train FNN'
model_FNN =  'model_best_res50.pt'   #'model file from trian FNN'
correct = 0
total = 1500 #total imaga of testset
inside = 0
total_time = 0

link_test_folder = 'F:\\KD_for_place365\\final_dataset\\test' # path testset
all_folders = list(sorted(glob.glob(f'{link_test_folder}\\*')))
for folder in all_folders:
  t1 = time.time()
  inside += 1
  inside_correct = 0
  label = folder.split('\\')[-1]
  urls_img = list(sorted(glob.glob(f'{folder}\\*.png')))
  for url in urls_img:
    predicted = predict(url, model_, model_place365, labels_url, model_FNN)
    if predicted == label:
      correct += 1
      inside_correct += 1
  t2 = time.time()
  intime = t2 - t1
  total_time += intime
  print('{}/30 complete {}, acc: {}, total time: {}'.format(inside, label, inside_correct/50, intime))
print('Accuracy of the network on the testset images: {} , total time {}'.format(100 * correct / total, total_time))