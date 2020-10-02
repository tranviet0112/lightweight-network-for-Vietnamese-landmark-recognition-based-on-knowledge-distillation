import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from tqdm import tqdm
import os 
import data_loader
import net_distill
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=None, help="Directory for the dataset")
parser.add_argument('--student_model', default=None, help="3conv or 5conv or 7conv")
parser.add_argument('--name_model', default='', help="name_of_output_file")
parser.add_argument('--epochs', default = 50, help='default 50')
parser.add_argument('--lr', default = 1e-3, help='learning rate with default 1e-3')

def train(feature_dir, epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, classes = data_loader.data_loader(args.data_dir)
    # write classes to txt
    with open('labels_{}.txt'.format(args.name_model), 'w') as f:
        for item in classes:
            f.write("%s\n" % item)
    if args.student_model == '3conv':       
        model = net_distill.Net_3conv(num_channels = 16, num_classes = len(classes))
    if args.student_model == '5conv':       
        model = net_distill.Net_5conv(num_channels = 16, num_classes = len(classes))
    if args.student_model == '7conv':       
        model = net_distill.Net_7conv(num_channels = 16, num_classes = len(classes))
    model.to(device)
    learning_rate = eval(args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    n_epochs = epochs # you may increase this number to train a final model

    valid_loss_min = np.Inf # track change in validation loss
    best_score = 0.0
    best_loss = 1e18

    for epoch in tqdm(range(1, n_epochs + 1)):
        # model_index = 0
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            # if train_on_gpu:
            data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################

        correct = 0
        total = 0

        model.eval()
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            # if train_on_gpu:
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        accuracy = correct/total*100

        # calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)

        # print training/validation statistics
        if epoch %10 == 0:
            print('Epoch: {} \tTraining Loss: {:.2f} \tValidation Loss: {:.2f}'.format(
                epoch, train_loss, valid_loss))
            print('Accuracy on epoch {}: {:.2f}'.format(epoch, accuracy))
        # save model if validation loss has decreased
        if best_loss >=valid_loss and best_score <= accuracy:
            print('Validation loss decreased ({:.2f} --> {:.2f}).  Saving model ...'.format(
                best_loss,
                valid_loss))
            model_file_path = os.path.join('model_best_{}.pt'.format(args.name_model))
            torch.save(model.state_dict(), model_file_path)
            best_loss = valid_loss
            best_score = accuracy

if __name__ == '__main__':

    # load the model
    args = parser.parse_args()
    train(args.data_dir, int(args.epochs))
    