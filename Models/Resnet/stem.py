import torch
import torch.nn as nn

class Stem(nn.Module):

    def __init__(self,in_channels,out_channels):
        super(Stem,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels,out_channels,(7,7),(2,2),(3,3),bias=False) #input = 224, output = 112, stride = 2x2, kernel_size = 7x7.thus padding = 3x3 
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d((3,3),(2,2),padding=1) #input = 112, output = 56,stride = 2x2,kernel_size = 3x3,hence padding = 1x1

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        return x