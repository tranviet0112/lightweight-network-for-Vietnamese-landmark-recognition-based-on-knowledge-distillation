import torch.nn as nn
import torch.nn.functional as F

class MyFNN(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(MyFNN,self).__init__()
        self.D_in = D_in
        self.H1 = H1
        self.H2 = H2
        self.D_out = D_out

        self.layer1 = nn.Linear(self.D_in, self.H1)
        self.layer2 = nn.Linear(self.H1, self.H2)
        self.layer3 = nn.Linear(self.H2, self.D_out)

        self.softmax = nn.Softmax(dim=1)

        self.bn1 = nn.BatchNorm1d(D_in)
        self.bn2 = nn.BatchNorm1d(H1)
        self.bn3 = nn.BatchNorm1d(H2)
        self.bn4 = nn.BatchNorm1d(D_out)

    def forward(self,x):
        # x -> BNorm
        x_bn = self.bn1(x)
        # x -> BNorm -> Linear(1).ELU 
        h1_elu = F.elu(self.layer1(x_bn))
        # x -> BNorm -> Linear(1).ELU -> BNorm -> 
        h1_elu_bn = self.bn2(h1_elu)
        # x -> BNorm -> Linear(1).ELU -> BNorm -> Linear(2).ELU 
        h2_elu = F.elu(self.layer2(h1_elu_bn))
        #  x -> BNorm -> Linear(1).ELU -> BNorm -> Linear(2).ELU  -> BNorm
        h2_elu_bn = self.bn3(h2_elu)
        # x -> BNorm -> Linear(1).ELU -> BNorm -> Linear(2).ELU  -> BNorm -> Linear(3).ELU
        out = F.elu(self.layer3(h2_elu_bn))
        # # x -> BNorm -> Linear(1).ELU -> BNorm -> Linear(2).ELU  -> BNorm -> Linear(3).ELU -> BNorm
        out_bn = self.bn4(out)

        output = self.softmax(out_bn)
        return output

def FNN():
    return MyFNN()