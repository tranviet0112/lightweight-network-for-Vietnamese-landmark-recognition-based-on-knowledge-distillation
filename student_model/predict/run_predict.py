import torch
import torchvision
import torchvision.transforms as transforms
import glob
from predict import predict
import time
# prepare
model_distil_CNN = 'path of student model weight'
labels_url = 'path labels from FNN'
model_FNN =  'path weight from FNN'
studend_model, num_classes_extract_step, num_channels = "5CNN", '30 for mode C and 365 for mode B', 16

#####
correct = 0
total = 1500 # total image of testset
inside = 0
total_time = 0

link_test_folder = 'paht of folder testset'
all_folders = list(sorted(glob.glob(f'{link_test_folder}\\*')))
for folder in all_folders:
  t1 = time.time()
  inside += 1
  inside_correct = 0
  label = folder.split('\\')[-1]
  urls_img = list(sorted(glob.glob(f'{folder}\\*.png')))
  for url in urls_img:
    predicted = predict(url, model_distil_CNN, studend_model, num_classes_extract_step, num_channels, labels_url, model_FNN)
    if predicted == label:
      correct += 1
      inside_correct += 1
  t2 = time.time()
  intime = t2 - t1
  total_time += intime
  print('{}/30 complete {}, acc: {}, total time: {}'.format(inside, label, incorrect/50, intime))
print('Accuracy of the network on the testset images: {} , total time {}'.format(100 * correct / total, total_time))