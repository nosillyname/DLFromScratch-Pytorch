import torch
import torch.nn as nn
import torch.nn.functional as F
from ...Utils.cnn_utils import get_cnn_output_size


class LeNet5(nn.Module):
    def __init__(self, in_channels, num_classes, input_size=(32, 32)):
        super(LeNet5, self).__init__()
        
        # Convolution + pooling layers
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1)   # 6x(H-4)x(W-4)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)            # 16x(H-8)x(W-8)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1)

        # --- Dynamically compute flatten size ---
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_size)  # batch=1
            out = self._forward_features(dummy)
            flatten_dim = out.view(1, -1).size(1)             # number of features
        
        # Fully connected layers
        self.fc1 = nn.Linear(flatten_dim, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def _forward_features(self, x):
        """Forward pass up to conv3 (feature extractor)."""
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, 1)   # flatten all but batch
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = LeNet5(1,10)
x = torch.randn(1,1,32,32)
print(model(x).shape) #testing the model output, should be batch_size * 10

