import torch
import torch.nn as nn
from conv import Conv2dBN,Conv2dBNRelu

#Reusable Residual Block for Resnet(18/32)
class ResidualBlockSmall(nn.Module):
    expansion = 1  
    def __init__(self,in_channels,out_channels,stride = 1,downsample_flag : bool = False):
        super(ResidualBlockSmall,self).__init__()

        self.conv1 = Conv2dBNRelu(in_channels,out_channels,k_size = 3,stride = stride,padding=1)
        self.conv2 = Conv2dBN(out_channels,out_channels*ResidualBlockSmall.expansion,k_size = 3,stride = 1,padding=1)
        self.relu = nn.ReLU(inplace=True)
        if downsample_flag:
            self.downsample = nn.Conv2d(in_channels,out_channels*ResidualBlockSmall.expansion,kernel_size=1,stride=stride)
            self.bn = nn.BatchNorm2d(out_channels*ResidualBlockSmall.expansion)
        else:
            self.downsample = None
            self.bn = None # for consistensy

    def forward(self,x):
        i = x

        x = self.conv1(x)
        x = self.conv2(x)

        if self.downsample is not None:
            i = self.downsample(i)
            i = self.bn(i)

        x += i
        x = self.relu(x)

        return x


#Reusable Residual Block for Resnet(50/101/152)
class ResidualBlockLarge(nn.Module):
    expansion = 4
    def __init__(self,in_channels,out_channels,stride,downsample_flag:bool = False):
        super(ResidualBlockLarge,self).__init__()   
   
        self.conv1 = Conv2dBNRelu(in_channels,out_channels,k_size=1,stride = 1,padding=0)
        self.conv2 = Conv2dBNRelu(out_channels,out_channels,k_size = 3,stride = stride,padding=1)
        self.conv3 = Conv2dBN(out_channels,out_channels*ResidualBlockLarge.expansion,k_size = 1,stride = 1,padding=0) # 4 times the filters
        self.relu = nn.ReLU(inplace=True)
        if downsample_flag:
            self.downsample = nn.Conv2d(in_channels,out_channels*ResidualBlockLarge.expansion,kernel_size=1,stride=stride,bias=False)
            self.bn = nn.BatchNorm2d(out_channels*ResidualBlockLarge.expansion)
        else:
            self.downsample = None
            self.bn = None # for consistensy
    
    def forward(self,x):
        i = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.downsample is not None:
            i = self.downsample(i)
            i = self.bn(i)
        x += i
        x = self.relu(x)

        return x
    