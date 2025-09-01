import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class LeNet5(nn.Module):

    def __init__(self,in_channels,num_classes):
        super(LeNet5,self).__init__()   
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=6,kernel_size=(5,5),stride=(1,1)) #INPUT: 32x32x1 ; OUTPUT: 6x28x28
        self.sub_samp = nn.AvgPool2d(kernel_size=(2,2),stride=(2,2)) #INPUT: 6x28x28 ; OUTPUT: 6x14x14
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5),stride=(1,1)) #INPUT: 6x14x14 ; OUTPUT: 16x10x10
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=(5,5),stride=(1,1)) #INPUT:16x5x5 ; OUTPUT: 120x1x1
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,num_classes)

    def forward(self,x):
        x = f.relu(self.conv1(x)) #original paper used TanH, I prefer relu
        x = self.sub_samp(x)
        x = f.relu(self.conv2(x))
        x = self.sub_samp(x)
        x = f.relu(self.conv3(x))
        x = x.reshape(x.shape[0],-1)
        x = f.relu(self.fc1(x))
        return self.fc2(x)
    
model = LeNet5(1,10)
x = torch.randn(1,1,32,32)
print(model(x).shape) #testing the model output, should be batch_size * 10

