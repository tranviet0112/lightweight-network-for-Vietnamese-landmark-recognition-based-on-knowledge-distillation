# Lightweight Network for Vietnamese Landmark Recognition based on Knowledge Distillation
## Install
* Clone the repo
  ```
  git clone https://github.com/ngocthinh248/Lightweight-Network-for-Vietnamese-Landmark-Recognition-based-on-Knowledge-Distillation.git
  ```

* Install the dependencies (including Pytorch)
  ```
  pip install -r requirements.txt
  ```
* Link for Dataset
  ```
  VN_Landmark_dataset:
  Train https://bom.to/9uY22bj
  Test  https://bom.to/na8RH7V
  Subset of Places365 dataset for Model B
  https://bom.to/na8RH7V
  
  ```
## Step 1 : training teacher/student model

Run train.py in knowledge-distillation-pytorch folder
```
change working directory to knowledge-distillation-pytorch folder
!python train.py --mode C (Choos which model you want to implement) --num_classes 30 (Choose number of class --model_teacher resnet50 --model_student 5CNN --model_dir ..\knowledge-distillation-pytorch\experiments\cnn_distill (directory contain params.json file) --data_dir directory of data use to train

```
## Step 2: Extract Feature
change working directory to extract_feature folder
```
!python run_placesCNN_unified.py --data_dir path to directory of dataset need to extract feature  --model_dir directory contain weight file from step 1
                                 --root_output path to directory for output feature  --student_model 5CNN   --num_classes 30  
                                 --num_channels 16 param for student model architecture
                                 
```
## Step 3: Train FNN (Feedforward Neural Network)
change working directory to FNN folder
```
!python train.py --feature_dir Directory for the dataset  --name_model name for the output file --epochs 100

```
## Step 4: Predict on test set

change working directory to predict folder
Modify this parameter in run_predict.py follow your setting before run
model_distil_CNN = 'path of student model weight'
labels_url = 'path labels from FNN'
model_FNN =  'path weight from FNN'
studend_model, num_classes_extract_step, num_channels = "5CNN", '30 for mode C and 365 for mode B', 16

```
python! run_predict.py

```
