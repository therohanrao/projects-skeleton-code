import torch
import torch.nn as nn
import torch.nn.functional as F


class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 5, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(5, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 56 * 56, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 5)

        # self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()

    def forward(self, x):

        # (n, 3, 224, 224)
        x = self.conv1(x)
        x = F.relu(x)
        # (n, 5, 224, 224)

        x = self.pool(x)
        # (n, 5, 112, 112)

        x = self.conv2(x)
        x = F.relu(x)
        # (n, 8, 112, 112)

        x = self.pool(x)
        # (n, 8, 56, 56)

        x = torch.reshape(x, (-1, 8 * 56 * 56))
        # (n, 8 * 56 * 56)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # (n, 256)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # (n, 128)

        x = self.fc3(x)
        # (n, 5)
        return x


class StartingNetwork2(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
        self.model_a = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained = True)
        self.fc = torch.nn.Sequential(*(list(self.model_a.children())[:-1]))
        self.model_a = self.fc
        #print(self.model_a)

        test = torch.rand(32,3,224,224)

        print(self.model_a(test).size())

        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 5)
  

    def forward(self, x):
        #with torch.no_grad():
        features = self.model_a(x)
        features = features.reshape(-1, 512)
        x = self.fc1(features)
        x = self.bn1(x)
        x = F.relu(x)
        # (n, 256)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # (n, 128)

        x = self.fc3(x)
        # (n, 5)
        return x
