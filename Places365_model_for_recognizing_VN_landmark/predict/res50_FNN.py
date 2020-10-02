import torch.nn as nn
import torch.nn.functional as F
import torch
class MyFNN(nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(MyFNN,self).__init__()
        self.D_in = D_in
        self.H1 = H1
        self.H2 = H2
        self.H3 = H3
        self.D_out = D_out

        self.layer1 = nn.Linear(self.D_in, self.H1)
        self.layer2 = nn.Linear(self.H1, self.H2)
        self.layer3 = nn.Linear(self.H2, self.H3)
        self.layer4 = nn.Linear(self.H3, self.D_out)
        self.softmax = nn.Softmax(dim=1)

        self.bn1 = nn.BatchNorm1d(D_in)
        self.bn2 = nn.BatchNorm1d(H1)
        self.bn3 = nn.BatchNorm1d(H2)
        self.bn4 = nn.BatchNorm1d(H3)
        self.bn5 = nn.BatchNorm1d(D_out)

    def forward(self,x):
        # x -> BNorm
        x = self.bn1(x)
        # x -> BNorm -> Linear(1).ELU 
        x = self.bn2(F.elu(self.layer1(x)))

        x = self.bn3(F.elu(self.layer2(x)))
    
        x = self.bn4(F.elu(self.layer3(x)))

        x = self.bn5(F.elu(self.layer4(x)))

        x = self.softmax(x)

        return x
