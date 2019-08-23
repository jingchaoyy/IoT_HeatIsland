"""
Created on 8/22/2019
@author: no281
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset,TensorDataset
import numpy as np
from Temp_pred_lpy.data_helper import *

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print('device using:',device)

# Hyper-parameters
input_size = 100
hidden_size1 = 50
hidden_size2 = 20
num_classes = 1
num_epochs = 8
batch_size = 16
learning_rate = 0.003

# MNIST dataset
# train_dataset = torchvision.datasets.MNIST(root='data',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)
#
# test_dataset = torchvision.datasets.MNIST(root='data',
#                                           train=False,
#                                           transform=transforms.ToTensor())

train_x,train_y,test_x,test_y=gen_train_and_test_data(shuffle=True,
                                                      cut_bin=False,
                                                      y_is_percentage=False)
train_x = torch.from_numpy(train_x[:train_x.shape[0]-train_x.shape[0]%batch_size])
train_y_plot = train_y
train_y = torch.from_numpy(train_y[:train_x.shape[0]-train_x.shape[0]%batch_size])
test_x_plot = test_x
test_x = torch.from_numpy(test_x[:test_x.shape[0]-test_x.shape[0]%batch_size])
test_y_plot = test_y
test_y = torch.from_numpy(test_y[:test_x.shape[0]-test_x.shape[0]%batch_size])

train_dataset = TensorDataset(train_x,train_y)
test_dataset = TensorDataset(test_x,test_y)
# Data loader
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


model = NeuralNet(input_size, hidden_size1, hidden_size2, num_classes).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.reshape(-1, 100).to(device).float()
        labels = labels.reshape(batch_size,1).to(device).float()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 1 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

with torch.no_grad():
    pred = None
    # 改变predict时候使用的是train数据还是test数据
    for images, labels in test_loader:
        images = images.reshape(-1, 100).to(device).float()
        labels = labels.to(device).float()
        outputs = model(images)
        if pred is None:
            pred = outputs.numpy()
        else:
            pred = np.append(pred,outputs.numpy())
    print(pred)

# plot_results(pred,test_y_plot)
plot_results_multiple(test_x_plot,test_y_plot,20,model,'torch_dnn')
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.reshape(-1, 100).to(device).float()
#         labels = labels.to(device).float()
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item().long()
#
#     print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'DNN-model.ckpt')