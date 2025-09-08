import torch
import torch.nn as nn
from skip_blocks import ResidualBlockLarge,ResidualBlockSmall
from stem import Stem


"""
Some notes before implementing the architechture:
    for the same output feature map size, the layers have the same number of filters;
    if the feature map size is halved, the number of filters is doubled so as to preserve the time complexity per layer;
    Formally, in this paper we consider a building block defined as: y = F(x, {Wi}) + x if dimensions of F and x are equal
    if they're not equal, project linearly y = F(x, {Wi}) + Ws*x
    We perform downsampling directly by convolutional layers that have a stride of 2.
    We adopt batch normalization (BN) [16] right after each convolution and before activation
    ReLu is used 
    Resnet 18/32 seem to have 2xConv_x layers of same size and bigger ones(Resnet 50 and above) have Conv layers that increase the filters by 4 times,so I'll implement them separately
    Paper seems to mention something like this:
        Conv2d->Batch-Norm->Relu
"""


    
class ResNet(nn.Module):

    def __init__(self,in_channels,out_classes,block,num_blocks:list,channels):
        super(ResNet,self).__init__()

        self.in_channels = in_channels

        #stem
        self.stem = Stem(in_channels,channels[0])

        self.in_channels = 64
        
        #skipnets 
        #Downsampling is performed by conv3 1, conv4 1, and conv5 1 with a stride of 2.
        self.conv2_x = self._create_layer(block, channels[0],num_blocks[0],stride = (1,1))
        self.conv3_x = self._create_layer(block, channels[1],num_blocks[1],stride = 2)
        self.conv4_x = self._create_layer(block, channels[2],num_blocks[2],stride = 2)
        self.conv5_x = self._create_layer(block, channels[3],num_blocks[3],stride = 2)

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels,out_classes)



    
    def _create_layer(self,block:ResidualBlockSmall,out_channels,num_blocks,stride):
         
        layers = []

        downsample_flag = stride!=1 or self.in_channels != out_channels*block.expansion

        layers.append(block(self.in_channels,
                            out_channels,
                            stride,
                            downsample_flag))
        

        #in channels for next block
        self.in_channels = out_channels * layers[0].expansion

        #setting up the blocks
        for _ in range(1,num_blocks):
            layers.append(block(self.in_channels,
                           out_channels,
                           stride=1,
                           downsample_flag=False))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.stem(x) #mentioned as conv1
        print(f"shape after stem {x.shape}")
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x