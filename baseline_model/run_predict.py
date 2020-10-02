from predict import predict
import glob
import time

labels_url = 'labels_MODEL_A_3conv.txt' #path of label output from training '
model_A =      'model_best_MODEL_A_3conv.pt'            #'path of weight output from training'
student_model =   '3conv'       #'type of student_model: 3conv or 5conv or 7conv'
correct = 0
total = 1500 # total image of testset
inside = 0
total_time = 0
link_test_folder = 'F:\\KD_for_place365\\final_dataset\\test'
all_folders = list(sorted(glob.glob(f'{link_test_folder}\\*')))

for folder in all_folders:
  t1 = time.time()
  inside += 1
  inside_correct = 0
  label = folder.split('\\')[-1]
  urls_img = list(sorted(glob.glob(f'{folder}\\*.png')))
  for url in urls_img:
    predicted = predict(url, labels_url, student_model, model_A)
    if predicted == label:
      correct += 1
      inside_correct += 1
  t2 = time.time()
  intime = t2 - t1
  total_time += intime
  print('{}/30 complete {}, acc: {}, total time: {}'.format(inside, label, inside_correct/50, intime))
print('Accuracy of the network on the testset images: {} , total time {}'.format(100 * correct / total, total_time))