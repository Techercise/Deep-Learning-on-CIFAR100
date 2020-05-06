# CISC642 PR3 part 2 - Matthew Leinhauser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

# create the training and test sets

dataset_root = './data'

# resize the images to be the same size, convert them to be Tensor Objects, and
# normalize the tensor values based on the mean and standard deviation of the
# RGB values

data_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5],
                                                           [0.5, 0.5, 0.5])])

trainset = torchvision.datasets.CIFAR100(dataset_root, train=True, transform =
                                         data_transforms, download=True)
testset = torchvision.datasets.CIFAR100(dataset_root, train=False, transform =
                                        data_transforms, download=True)

# create a data loader for both the training set and testing set
train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size =16, shuffle=False)

# Create the model
class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.layer1 = nn.Sequential(
        nn.Conv2d(3, 20, 5, 1, 2),
        nn.ReLU(),
        nn.MaxPool2d(2, 4))
    self.layer2 = nn.Sequential(
        nn.Conv2d(20, 40, 5, 1, 2),
        nn.ReLU(),
        nn.MaxPool2d(2, 2))
    self.layer3 = nn.Sequential(
        nn.Conv2d(40, 80, 5, 1, 2),
        nn.ReLU(),
        nn.MaxPool2d(2, 2))
    # add a dropout layer to avoid overfitting
    self.drop_out = nn.Dropout()
    self.fully_con_1 = nn.Linear(14 * 14 * 80, 200)
    self.fully_con_2 = nn.Linear(200, 100)

  def forward(self, x):
    output = self.layer1(x)
    output = self.layer2(output)
    output = self.layer3(output)
    output = output.reshape(output.size(0), -1)
    output = self.drop_out(output)
    output = self.fully_con_1(output)
    output = self.fully_con_2(output)
    return output

# Set up the model

# hyperparameters
num_epochs = 30
learning_rate = 0.001

# set up the model for training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model
# iterate over the mini-batchs and set the optimizer as zero_grad(). Then use
# the current model weight for predication and backpropogating the prediction
# loss

for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)

    # Forward Pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Optimize Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Track the accuracy and print out where we are in the training process
    total = labels.size(0)
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    if(i + 1) % 1000 == 0:
      print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
      .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), 
              (correct/ total) * 100))

# Save the model weights
model_weights = copy.deepcopy(model.state_dict())
torch.save(model_weights, 'best_model_weight.pth')

# Test the model
num_correct = 0
num_samples = 0

model.load_state_dict(torch.load('best_model_weight.pth'))

model.eval()
for epoch in range(num_epochs):
# with torch.no_grad():
  correct = 0
  total = 0
  for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    num_samples += labels.size(0)
    num_correct += (preds == labels).sum().item()
  
print('Accuracy: ', (num_correct/num_samples) * 100, '%')
