import torch
import torch.nn as nn
import torch.nn.functional as F

# dwt1 image shape: (128, 128, 75)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convs, ? in, ? out, ?x? kernel
        self.conv1 = nn.Conv3d(1, 4, 3)
        self.conv2 = nn.Conv3d(4, 8, 3)
        self.conv3 = nn.Conv3d(8, 16, 3)
        self.conv4 = nn.Conv3d(16, 32, 3)
        # fully connected layers
        self.fc1 = nn.Linear(32*6*6*2, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        # x shape (N, 1, 128, 128, 75)
        x = F.max_pool3d(F.relu(self.conv1(x)), 2)  # x shape (4, 63, 63, 36) <-- (4, 126, 126, 73)
        x = F.max_pool3d(F.relu(self.conv2(x)), 2)  # x shape (8, 30, 30, 17)   <-- (8, 61, 61, 34)
        x = F.max_pool3d(F.relu(self.conv3(x)), 2)  # x shape (16, 14, 14, 7)  <-- (16, 28, 28, 15)
        x = F.max_pool3d(F.relu(self.conv4(x)), 2)  # x shape (32, 6, 6, 2)   <-- (32, 12, 12, 5)
        x = x.view(x.shape[0], -1)                  # x shape (1600,)
        x = F.relu(self.fc1(x))                     # x shape (600,)
        x = F.relu(self.fc2(x))                     # x shape (100,)
        x = F.relu(self.fc3(x))                     # x shape (10,)
        x = torch.sigmoid(self.fc4(x))                     # x shape (1,)
        return x

# net = Net()
# print(net)
# params = list(net.parameters())
# print(len(params))
# print(params[1].size())  # conv1's .weight