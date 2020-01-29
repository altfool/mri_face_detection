import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convs, ? in, ? out, ?x? kernel
        self.conv1 = nn.Conv3d(1, 4, 7)
        self.conv2 = nn.Conv3d(4, 8, 5)
        self.conv3 = nn.Conv3d(8, 16, 3)
        self.conv4 = nn.Conv3d(16, 32, 3)
        self.conv5 = nn.Conv3d(32, 64, 3)
        # fully connected layers
        self.fc1 = nn.Linear(64*5*5*2, 600)
        self.fc2 = nn.Linear(600, 100)
        self.fc3 = nn.Linear(100, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        # x shape (N, 1, 256, 256, 150)
        x = F.max_pool3d(F.relu(self.conv1(x)), 2)  # x shape (4, 125, 125, 72) <-- (4, 250, 250, 144)
        x = F.max_pool3d(F.relu(self.conv2(x)), 2)  # x shape (8, 60, 60, 34)   <-- (8, 121, 121, 68)
        x = F.max_pool3d(F.relu(self.conv3(x)), 2)  # x shape (16, 29, 29, 16)  <-- (16, 58, 58, 32)
        x = F.max_pool3d(F.relu(self.conv4(x)), 2)  # x shape (32, 13, 13, 7)   <-- (32, 27, 27, 14)
        x = F.max_pool3d(F.relu(self.conv5(x)), 2)  # x shape (64, 5, 5, 2)     <-- (64, 11, 11, 5)
        x = x.view(x.shape[0], -1)                  # x shape (3200,)
        x = F.relu(self.fc1(x))                     # x shape (600,)
        x = F.relu(self.fc2(x))                     # x shape (100,)
        x = F.relu(self.fc3(x))                     # x shape (10,)
        x = F.sigmoid(self.fc4(x))                     # x shape (1,)
        return x

# net = Net()
# print(net)
# params = list(net.parameters())
# print(len(params))
# print(params[1].size())  # conv1's .weight