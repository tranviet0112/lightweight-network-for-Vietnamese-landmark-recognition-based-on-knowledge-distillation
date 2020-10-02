import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net_3conv(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions

    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, num_channels, num_classes):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        Args:
            params: (Params) contains num_channels
        """
        super(Net_3conv, self).__init__()
        self.num_channels = num_channels    #old self.num_channels = params.num_channels 
        self.num_classes = num_classes
        # thiết kế mạng ở đây

        self.conv1 = nn.Conv2d(3, self.num_channels*2, 4, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(self.num_channels*2)
        # sau khi qua pool2d(2) = 114

        self.conv2 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 4, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(self.num_channels*4)
        # pool2d(2) = 28
        

        self.conv3 = nn.Conv2d(self.num_channels*4, self.num_channels*8, 4, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(self.num_channels*8)
        #pool = 12

        self.avgpool = nn.AvgPool2d(6)

        self.bnin = nn.BatchNorm1d(512)
        # 2 fully connected layers to transform the output of the convolution layers to the final output
        
        self.fc1 = nn.Linear(512, 256)
        self.bnfc1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.bnfc2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, self.num_classes)
        self.bnfc3 = nn.BatchNorm1d(self.num_classes)
    
              
        self.softmax = nn.Softmax(dim=1)
              
    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 32 x 32 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        #                                                  -> batch_size x 3 x 224 x 224, 
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = self.bn1(self.conv1(s))                         # batch_size x num_channels*1 x 222 x 222
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*1 x 111 x 111

        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 108 x 108
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 54 x 54

        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*3 x 52 x 52 
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*3 x 26 x 26, 
        
        s = self.avgpool(s)                                 # batch_size x num_channels*8 x 2 x 2
        # flatten the output for each image   
        s = self.bnin(s.view(-1, 2*2*self.num_channels*8))             # batch_size x 2*2*num_channels*8 
     
        s = F.elu(self.bnfc1(self.fc1(s)))                                     # batch_size x 30
        s = F.elu(self.bnfc2(self.fc2(s)))
        s = F.elu(self.bnfc3(self.fc3(s)))

        s = self.softmax(s)
        return s


class Net_5conv(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions

    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, num_channels, num_classes): 
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        Args:
            params: (Params) contains num_channels
        """
        super(Net_5conv, self).__init__()
        self.num_channels = num_channels    #old self.num_channels = params.num_channels 
        self.num_classes = num_classes
        # thiết kế mạng ở đây

        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        # sau khi qua pool2d(2) = 111

        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 4, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        # pool2d(2) = 54
        

        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)
        #pool = 26

        self.conv4 = nn.Conv2d(self.num_channels*4, self.num_channels*6, 3, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(self.num_channels*6)
        # pool = 14

        self.conv5 = nn.Conv2d(self.num_channels*6, self.num_channels*8, 3, stride=1, padding=2)
        self.bn5 = nn.BatchNorm2d(self.num_channels*8)
        # batch x 4x4 x
        self.avgpool = nn.AvgPool2d(4)

        self.bnin = nn.BatchNorm1d(512)
        # 2 fully connected layers to transform the output of the convolution layers to the final output
        
        self.fc1 = nn.Linear(512, 256)
        self.bnfc1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.bnfc2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, self.num_classes)
        self.bnfc3 = nn.BatchNorm1d(self.num_classes)
    
              
        self.softmax = nn.Softmax(dim=1)

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 32 x 32 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        #                                                  
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = self.bn1(self.conv1(s))                         
        s = F.relu(F.max_pool2d(s, 2))                      
        s = self.bn2(self.conv2(s))                         
        s = F.relu(F.max_pool2d(s, 2))                      

        s = self.bn3(self.conv3(s))                          
        s = F.relu(F.max_pool2d(s, 2))                       

        s = self.bn4(self.conv4(s))                         
        s = F.relu(F.max_pool2d(s, 2))                      

        s = self.bn5(self.conv5(s))                         
        s = F.relu(F.max_pool2d(s, 2))                      
        
        s = self.avgpool(s)                                 
        # flatten the output for each image   
        s = self.bnin(s.view(-1, 2*2*self.num_channels*8))              
     
        s = F.elu(self.bnfc1(self.fc1(s)))                                    
        s = F.elu(self.bnfc2(self.fc2(s)))
        s = F.elu(self.bnfc3(self.fc3(s)))

        s = self.softmax(s)

        return s


class Net_7conv(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions

    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, num_channels, num_classes): #old def __init__(self, params) 
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        Args:
            params: (Params) contains num_channels
        """
        super(Net_7conv, self).__init__()
        self.num_channels = num_channels    #old self.num_channels = params.num_channels 
        self.num_classes = num_classes
        # thiết kế mạng ở đây

        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        # sau khi qua pool2d(2) = 111

        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 4, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
        # pool2d(2) = 56
        

        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=3)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)
        #pool = 30

        self.conv4 = nn.Conv2d(self.num_channels*4, self.num_channels*6, 3, stride=1, padding=2)
        self.bn4 = nn.BatchNorm2d(self.num_channels*6)
        # pool = 16

        self.conv5 = nn.Conv2d(self.num_channels*6, self.num_channels*8, 3, stride=1, padding=2)
        self.bn5 = nn.BatchNorm2d(self.num_channels*8)
        # pool = 9
        
        self.conv6 = nn.Conv2d(self.num_channels*8, self.num_channels*16, 3, stride=1, padding=2)
        self.bn6 = nn.BatchNorm2d(self.num_channels*16)
        # pool = 5

        self.conv7 = nn.Conv2d(self.num_channels*16, self.num_channels*32, 2, stride=1, padding=2)
        self.bn7 = nn.BatchNorm2d(self.num_channels*32)
        # batch x 4x4 x

        self.avgpool = nn.AvgPool2d(4)

        self.bnin = nn.BatchNorm1d(512)
        # 2 fully connected layers to transform the output of the convolution layers to the final output
        
        self.fc1 = nn.Linear(512, 256)
        self.bnfc1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.bnfc2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, self.num_classes)
        self.bnfc3 = nn.BatchNorm1d(self.num_classes)
    
              
        self.softmax = nn.Softmax(dim=1)

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 32 x 32 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        #                                                  -> batch_size x 3 x 224 x 224, 
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
        s = self.bn1(self.conv1(s))                         # batch_size x num_channels*1 x 222 x 222
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*1 x 111 x 111

        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 108 x 108
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 54 x 54

        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*3 x 52 x 52 
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*3 x 26 x 26, 

        s = self.bn4(self.conv4(s))                         # batch_size x num_channels*6 x 28 x 28
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*6 x 14 x 14

        s = self.bn5(self.conv5(s))                         # batch_size x num_channels*8 x 8 x 8
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*8 x 4 x 4

        s = self.bn6(self.conv6(s))                         # batch_size x num_channels*6 x 28 x 28
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*6 x 14 x 14

        s = self.bn7(self.conv7(s))                         # batch_size x num_channels*8 x 8 x 8
        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*8 x 4 x 4


        s = self.avgpool(s)                                 # batch_size x num_channels*8 x 2 x 2
        # flatten the output for each image   
        s = self.bnin(s.view(-1, 1*1*self.num_channels*32))             # batch_size x 2*2*num_channels*8 
     
        s = F.elu(self.bnfc1(self.fc1(s)))                                     # batch_size x 30
        s = F.elu(self.bnfc2(self.fc2(s)))
        s = F.elu(self.bnfc3(self.fc3(s)))

        s = self.softmax(s)
        return s