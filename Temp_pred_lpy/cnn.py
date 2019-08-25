"""
Created on 8/23/2019
@author: no281
"""
import sys
import torch
import json
import numpy as np
import pandas as pd
import os
from cnn_lstm import get_data
from Temp_pred_lpy.data_helper import *
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset,TensorDataset

sys.path.append(sys.path[0]+'/../')

device = torch.device('cpu')

train_x,train_y,test_x,test_y = gen_cnn_data(y_is_center_point=True)
input_channel_num = train_x.shape[1]

# 得到如下形状的数据：
# train_x.shape = (1736,99,7,7) test_x.shape = (225,99,7,7)
# train_y.shape = (1736,) test_y.shape = (225,)
print("data shape:%s %s %s %s" % (train_x.shape,train_y.shape,test_x.shape,test_y.shape))
print('data done!')

num_epochs = 8
batch_size = 16

train_dataset = TensorDataset(torch.from_numpy(train_x[:train_x.shape[0] - train_x.shape[0]%batch_size]),
                              torch.from_numpy(train_y[:train_x.shape[0] - train_x.shape[0] % batch_size]))
test_dataset = TensorDataset(torch.from_numpy(test_x[:test_x.shape[0]-test_x.shape[0]%batch_size]),
                             torch.from_numpy(test_y[:test_x.shape[0]-test_x.shape[0]%batch_size]))

train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


class CNN(nn.Module):
    def __init__(self,input_channels):
        super(CNN, self).__init__()
        self.origin_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=self.origin_channels,out_channels=self.origin_channels*2,kernel_size=5,padding=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=self.origin_channels*2,out_channels=self.origin_channels*2*2,kernel_size=2)
        self.fc1 = nn.Linear(in_features=self.origin_channels*2*2,out_features=120)
        self.fc2 = nn.Linear(in_features=120,out_features=30)
        self.fc3 = nn.Linear(in_features=30,out_features=1)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = x.view(-1,self.origin_channels*2*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = CNN(input_channel_num).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = images.to(device).float()
        labels = labels.to(device).float()

        outputs = model(images)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

with torch.no_grad():
    pred = None
    # 改变predict时候使用的是train数据还是test数据
    for images, labels in test_loader:
        images = images.to(device).float()
        labels = labels.to(device).float()
        outputs = model(images)
        if pred is None:
            pred = outputs.numpy()
        else:
            pred = np.append(pred,outputs.numpy())
    print(pred)

plot_results(pred,test_y)
# plot_results_multiple(test_x,test_y,20,model,'torch_dnn')