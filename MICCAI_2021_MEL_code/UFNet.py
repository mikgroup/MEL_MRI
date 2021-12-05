# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# class PixelNet(nn.Module):
#     def __init__(self,n_channels):
#         super(PixelNet, self).__init__()
#         self.conv1 = nn.Conv1d(n_channels,8, 5)
#         self.pool  = nn.MaxPool1d(2)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv1d(8, 16, 3)
#         self.conv3 = nn.Conv1d(16, 32, 3)
#         self.fc1   = nn.Linear(249*8*2, 512)
#         self.fc2   = nn.Linear(500, 256)
#         self.fc3   = nn.Linear(256, 64)
   
#         self.sigmoid = nn.Sigmoid()
# #         self.fc3   = nn.Linear(256, 1)
        
#     def forward(self, x):
# #         x1 = self.conv1(x)
# #         x2 = self.sigmoid(x1)
#         x3 = x.view(-1,500)
# #         x2 = self.pool(x1)
# #         x3 = self.relu(x2)
# #         x4 = self.conv2(x3)
# #         x5 = self.pool(x4)
# #         x6 = self.relu(x5)
# #         x7 = self.conv3(x6)
# #         x8 = self.pool(x7)
# #         x9 = self.relu(x8)
# #         x9 = x1.view(-1, 249*8*2)
# #         x10= self.fc1(x9)
# #         x11= self.relu(x10)
#         x4= self.fc2(x3)
#         x5= self.relu(x4)
#         x12 = self.fc3(x5)
#         norm1 = x12.norm(keepdim=True)
#         x13 = x12.div(norm1.expand_as(x12))
# #         x13= self.relu(x12)
# #         x14= self.fc3(x13)
        
#         return x13
    

# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# class PixelNet(nn.Module):
#     def __init__(self,n_channels):
#         super(PixelNet, self).__init__()
#         self.conv1 = nn.Conv1d(n_channels,32, 5)
#         self.pool  = nn.MaxPool1d(2)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv1d(32, 64, 3)
#         self.conv3 = nn.Conv1d(64, 128, 3)
#         self.fc1   = nn.Linear(60*128, 256)
#         self.fc2   = nn.Linear(256, 128)
#         self.fc3   = nn.Linear(128,22031)
#         self.fc4   = nn.Linear(496*32,22031)
#         self.sigmoid = nn.Sigmoid()
        
# #         self.fc3   = nn.Linear(256, 1)
        
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.relu(x1)
#         x3 = x2.view(-1, 496*32)
#         x4 = self.fc4(x3)
# #         x5 = 
# #         x2 = self.pool(x1)
# #         x3 = self.relu(x2)
# #         x4 = self.conv2(x3)
# #         x5 = self.pool(x4)
# #         x6 = self.relu(x5)
# #         x7 = self.conv3(x6)
# #         x8 = self.pool(x7)
# #         x9 = self.relu(x8)
# #         x9 = x9.view(-1, 60*128)
# #         x10= self.fc1(x9)
# #         x11= self.relu(x10)
# #         x12= self.fc2(x11)
# #         x13 = self.relu(x12)
# #         x14 = self.fc3(x13)
# #         x15 = self.sigmoid(x14)
# #         norm1 = x12.norm(keepdim=True)
# #         x13 = x12.div(norm1.expand_as(x12))
# #         x13= self.relu(x12)
# #         x14= self.fc3(x13)
        
#         return x4    
    
    
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import torch.nn.functional as f

class PixelNet(nn.Module):
    def __init__(self,n_channels,n_outputs):
        super(PixelNet, self).__init__()
        self.conv1 = nn.Conv1d(n_channels,32, 5)
        self.pool  = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, 5)
        self.conv3 = nn.Conv1d(64, 128, 5)
        self.fc1   = nn.Linear(59*128, 128)
        self.fc2   = nn.Linear(128, 32)
        self.fc3   = nn.Linear(32, n_outputs)
#         self.fc3   = nn.Linear(64, 7)
        self.sigmoid = nn.Sigmoid()
        
#         self.fc3   = nn.Linear(256, 1)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool(x1)
        x3 = self.relu(x2)
        x4 = self.conv2(x3)
        x5 = self.pool(x4)
        x6 = self.relu(x5)
        x7 = self.conv3(x6)
        x8 = self.pool(x7)
        x9 = self.relu(x8)
        x9 = x9.view(-1, 59*128)
        x10= self.fc1(x9)
        x11= self.relu(x10)
        x12= self.fc2(x11)
#         x13 = self.relu(x12)
        x13 = self.fc3(x12)
        x14 = f.normalize(x13, p=2, dim=1)
#         x13 = self.relu(x12)
#         x14 = self.fc3(x13)
#         x15 = self.sigmoid(x14)
#         norm1 = x12.norm(keepdim=True)
#         x13 = x12.div(norm1.expand_as(x12))
#         x13= self.relu(x12)
#         x14= self.fc3(x13)
        
        return x14