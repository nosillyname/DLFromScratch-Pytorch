import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

#network
class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,num_classes)

    def forward(self,x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#simple CNN
class CNN(nn.Module):
    def __init__(self,in_channels = 1, num_classes = 10):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1)) #same convolution,i.e same output size as input  
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=(1,1)) #same convolution,i.e same output size as input  
        self.fc1 = nn.Linear(16*7*7,num_classes)

    def forward(self,x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        
        return x

model = CNN(1,10)
x = torch.randn(64,1,28,28)
print(model(x).shape)

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#hyperparams
in_channels = 1
num_classes  =10
learning_rate = 1e-3
batch_size = 64
num_epochs = 10

#load data
train_dataset = datasets.MNIST(root = 'dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset=train_dataset,batch_size = batch_size,shuffle =  True)
test_dataset = datasets.MNIST(root = 'dataset/',train=False,transform=transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset=test_dataset,batch_size = batch_size,shuffle =  True)

# initialize network
model = CNN(in_channels= in_channels,num_classes=num_classes).to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

#Train Network
for epoch in tqdm(range(num_epochs)):
    for batch_idx, (data,targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        #flatten the goddamn images
        # data = data.reshape(data.shape[0],-1)

        #forward pass
        scores = model(data)
        loss = criterion(scores,targets)
        
        #backward
        optimizer.zero_grad()
        loss.backward()

        #optimizer step
        optimizer.step()


#check accuracy
def check_accruacy(loader,model):
    num_correct = 0
    num_samples = 0

    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            # x = x.reshape(x.shape[0],-1)

            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")   
        model.train()

# --- visualize learned filters ---
def visualize_filters(layer, num_filters=8):
    """
    Visualize filters (weights) from a Conv2d layer.
    """
    filters = layer.weight.data.clone().cpu()
    n = min(filters.shape[0], num_filters)  # number of filters to show
    fig, axs = plt.subplots(1, n, figsize=(15, 5))
    for i in range(n):
        # For conv1, just show single channel filters
        if filters.shape[1] == 1:
            axs[i].imshow(filters[i, 0, :, :], cmap="gray")
        else:
            # For conv2 (multi-channel), average across input channels
            avg_filter = filters[i].mean(dim=0)
            axs[i].imshow(avg_filter, cmap="gray")
        axs[i].axis("off")
    plt.suptitle("Filter weights")
    plt.show()

# --- visualize feature maps (activations) ---
def visualize_feature_maps(model, image):
    """
    Pass an image through conv1 and conv2, and visualize feature maps.
    """
    model.eval()
    with torch.no_grad():
        x1 = model.conv1(image.unsqueeze(0))  # (1,8,H,W)
        x1_act = nn.functional.relu(x1)
        x1_pooled = model.pool(x1_act)

        x2 = model.conv2(x1_pooled)           # (1,16,H/2,W/2)
        x2_act = nn.functional.relu(x2)

    # show conv1 activations
    fig, axs = plt.subplots(1, 8, figsize=(15, 5))
    for i in range(8):
        axs[i].imshow(x1_act[0, i].cpu(), cmap="gray")
        axs[i].axis("off")
    plt.suptitle("Conv1 activations")
    plt.show()

    # show conv2 activations
    fig, axs = plt.subplots(2, 8, figsize=(15, 5))
    for i in range(16):
        axs[i // 8, i % 8].imshow(x2_act[0, i].cpu(), cmap="gray")
        axs[i // 8, i % 8].axis("off")
    plt.suptitle("Conv2 activations")
    plt.show()


# --- run checks and visualizations ---
check_accruacy(test_loader, model)
check_accruacy(train_loader, model)

# visualize filters
visualize_filters(model.conv1, num_filters=8)
visualize_filters(model.conv2, num_filters=16)

# pick one test image
example_img, _ = next(iter(test_loader))
visualize_feature_maps(model, example_img[0].to(device))
