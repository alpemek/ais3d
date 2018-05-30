"""
Created on Mon Apr 30 17:54:54 2018

@author: chulekm
"""
#INCLUDING LIBRARIES
from __future__ import print_function
import torch

import torch.nn as nn
import torch.cuda as torchcuda

device = torchcuda.current_device()
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
torch.manual_seed(1)
import matplotlib.pyplot as plt
#device = torch.device("cuda")


#SOME GENERAL HYPERPARAMETERS
num_train_epochs = 20;
lr=0.01
momentum=0.5
#LOADING THE DATA
transform1=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]);
train_loader=torch.utils.data.DataLoader(datasets.ImageFolder('/home/chulekm/AutonomousDriving/Exercise 1/GTSRB/Training', transform=transform1),batch_size=64, shuffle=True, pin_memory=True)
validate_loader=torch.utils.data.DataLoader(datasets.ImageFolder('/home/chulekm/AutonomousDriving/Exercise 1/GTSRB/Validation',transform=transform1),batch_size=1, shuffle=True, pin_memory=True)
test_loader=torch.utils.data.DataLoader(datasets.ImageFolder('/home/chulekm/AutonomousDriving/Exercise 1/GTSRB/Testing',transform=transform1),batch_size=1, shuffle=True, pin_memory=True)
#CLASS SIMILAR TO VGG16
class Vgg16(nn.Module):
    def __init__(self, num_classes=43):
        super(Vgg16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),

            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),

        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

#CNN USED FOR REACHING OVER 90 PERCENT ACCURACY
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.conv3_drop = nn.Dropout2d()
        #self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(1024, 43)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2, stride=2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2, stride=2))
        x = x.view(-1,1024)
        #x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

#HERE ONE HAS TO SELECT ONE OF THE MODELS TO RUN THE SIMULATION
#model = Vgg16().to(device)
model = CNN().cuda(device)


optimizer = optim.SGD(model.parameters(), lr, momentum)
loss_Vector = list();
epoch_Vector = list();
accuracy_Vector = list();

#TRAINING OF THE MODEL
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()));
    loss_Vector.append(loss.item())
    epoch_Vector.append(epoch)

#VALIDATION OF THE MODEL
def validate():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validate_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item() # sum batch size
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(validate_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(validate_loader.dataset),
        100. * correct / len(validate_loader.dataset)))
    accuracy_Vector.append(100. * correct / len(validate_loader.dataset))

#TESTING OF THE MODEL
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#TRAINING THE MODEL FOR ALL THE EPOCHS
for epoch in range(1, num_train_epochs + 1):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    train(epoch)
    l1, = ax1.plot(epoch_Vector, loss_Vector, 'r', label='Loss')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel('Loss',color='r')
    validate()
    l2, = ax2.plot(epoch_Vector, accuracy_Vector, 'b', label='Accuracy')
    ax2.set_ylabel('Accuracy',color='b')
    ax1.legend(handles=[l1,l2])
    plt.show()

test()