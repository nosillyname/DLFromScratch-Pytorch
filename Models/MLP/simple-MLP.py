import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

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


model = NN(784,10)
x = torch.randn(64,784)
print(model(x).shape)

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#hyperparams
input_size = 784
num_classes  =10
learning_rate = 1e-3
batch_size = 64
num_epochs = 10

#load data
train_dataset = datasets.MNIST(root = 'dataset/',train=True,transform=transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset=train_dataset,batch_size = batch_size,shuffle =  True)
test_dataset = datasets.MNIST(root = 'dataset/',train=False,transform=transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset=train_dataset,batch_size = batch_size,shuffle =  True)

# initialize network
model = NN(input_size=input_size,num_classes=num_classes).to(device)

#Loss and optimizer
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

#Train Network
for epoch in tqdm(range(num_epochs)):
    for batch_idx, (data,targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        #flatten the goddamn images
        data = data.reshape(data.shape[0],-1)

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
            x = x.reshape(x.shape[0],-1)

            scores = model(x)
            _,predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")   
        model.train()
              
check_accruacy(test_loader,model)
check_accruacy(train_loader,model)