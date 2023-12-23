import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(512, 512)  
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(F.avg_pool2d(self.conv1(x), 2)) 
        x = F.relu(F.avg_pool2d(self.conv2(x), 2))  
        x = x.view(x.size()[0], -1) 
        x = F.relu(self.fc1(x))  
        x = self.fc2(x) 
        return x
    

class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(800, 512)  
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(F.avg_pool2d(self.conv1(x), 2)) 
        x = F.relu(F.avg_pool2d(self.conv2(x), 2))  
        x = x.view(x.size()[0], -1) 
        x = F.relu(self.fc1(x))  
        x = self.fc2(x) 
        return x


