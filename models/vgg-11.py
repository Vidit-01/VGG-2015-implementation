import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG(nn.Module):
    def __init__(self,num_classes=200):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,1) #224 x 224
        self.conv2 = nn.Conv2d(64,128,3,1) #
        self.conv3 = nn.Conv2d(128,256,3,1) #
        self.conv4 = nn.Conv2d(256,256,3,1) #
        self.conv5 = nn.Conv2d(256,512,3,1) #
        self.conv6 = nn.Conv2d(512,512,3,1) #
        self.conv7 = nn.Conv2d(512,512,3,1) #
        self.conv8 = nn.Conv2d(512,512,3,1) #

        self.fc1 = nn.Linear(512*6*6,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,200)


    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))

        x = torch.flatten(x,1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

