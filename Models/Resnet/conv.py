import torch
import torch.nn as nn

#Reusable Convolutional Blocks
class Conv2dBN(nn.Module):

    def __init__(self,in_channels,out_channels,k_size,stride,padding):
        super(Conv2dBN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,k_size,stride,padding,bias = False)
        self.bn = nn.BatchNorm2d(num_features=out_channels) #duh
        
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        return x

class Conv2dBNRelu(nn.Module):

    def __init__(self,in_channels,out_channels,k_size,stride,padding):
        super(Conv2dBNRelu,self).__init__()
        self.conv1 = Conv2dBN(in_channels,out_channels,k_size,stride,padding)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        return x
