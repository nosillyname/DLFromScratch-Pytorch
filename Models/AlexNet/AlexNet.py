import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    """
    Implementation of AlexNet architecture from the 2012 paper by Krizhevsky et al.
    Designed for image classification with input size (3, 227, 227).
    
    Args:
        in_channels (int): Number of input channels (default: 3 for RGB images).
        classes (int): Number of output classes (default: 1000 for ImageNet).
    """
    def __init__(self, in_channels=3, classes=1000):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=11, stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(6*6*256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, classes)

        self.norm = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.drop = nn.Dropout(p=0.5)

        self.initializeWeights()

    def forward(self, x):
        assert x.shape[2:] == (227, 227), "Input size must be (batch_size, 3, 227, 227)"
        x = F.relu(self.conv1(x))
        x = self.norm(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.norm(x)
        x = self.pool(x)   
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def initializeWeights(self):
        zero_bias_layers = {self.conv1, self.conv3, self.fc3}
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                if layer in zero_bias_layers:
                    nn.init.constant_(layer.bias, 0.0)
                else:
                    nn.init.constant_(layer.bias, 1.0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet().to(device)
x = torch.randn(32, 3, 227, 227).to(device)
print(model(x).shape)  # Should print: torch.Size([32, 1000])