# Lightweight Network for Vietnamese Landmark Recognition based on Knowledge Distillation
## Install
* Clone the repo
  ```
  git clone https://github.com/tranviet0112/lightweight-network-for-Vietnamese-landmark-recognition-based-on-knowledge-distillation
  ```

* Install the dependencies (including Pytorch)
  ```
  pip install -r requirements.txt
  ```
  
* Link for Dataset
  ```
  Vietnamese landmark dataset:
  Train https://bom.to/9uY22bj
  Test  https://bom.to/na8RH7V
  Subset of Places365 dataset:
  https://bom.to/na8RH7V
  ```
  
## Step 1 : train teacher/student

run train.py in knowledge_knowledge folder
```
change working directory to knowledge_knowledge folder

If training teacher/student on Subset of Places365 dataset:
!python train.py --mode B --num_classes 365 --model_teacher resnet18 --model_student 5CNN --model_dir experiments\cnn_distill (directory contain params.json file) --data_dir directory of data use to train

If training teacher/student on Vietnamese landmark dataset:
!python train.py --mode C --num_classes 30 --model_teacher resnet18 --model_student 5CNN --model_dir experiments\cnn_distill (directory contain params.json file) --data_dir directory of data use to train
```

## Step 2: extract features

change working directory to extract_feature folder
```
!python run_placesCNN_unified.py --data_dir directory path of dataset  --model_dir directory path contains weight file from step 1
                                 --root_output directory path of output (features of dataset)  --student_model 5CNN   --num_classes 30  
                                 --num_channels 16 param for student model architecture
                                 
```

## Step 3: train FNN (Feedforward Neural Network)

change working directory to FNN folder
```
!python train.py --feature_dir directory path of dataset  --name_model name for the output file --epochs 100

```

## Step 4: predict on test set

change working directory to predict folder
Modify this parameter in run_predict.py follow your setting before run
model_distil_CNN = 'path of student model weight'
labels_url = 'path labels from FNN'
model_FNN =  'path weight from FNN'
studend_model, num_classes_extract_step, num_channels = "5CNN", '30 for mode C and 365 for mode B', 16

```
!python run_predict.py

```
