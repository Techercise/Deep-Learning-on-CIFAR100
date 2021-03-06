# CISC642 - PR3 part 1 - Matthew Leinhauser

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

# load the VGG16 model wiith pretrained weights
model = models.vgg16(pretrained=True)

# extract the number of input features for the last fully connected model layer
num_features = model.classifier[6].in_features

# extract the first layer of the model
num_class = model.classifier[0].in_features

# replace the last fully connected layer with a new layer
model.classifier[6] = nn.Linear(num_features, num_class)

# freeze all layers except the last layer (output) because we don't want to
# update them
for param in model.parameters():
  param.requires_grad = False

for param in model.classifier[6].parameters():
  param.requires_grad = True

# hyperparameters
num_epochs = 10
learning_rate = 0.001

# set up the model for training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# TRAINING THE MODEL

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

# TESTING THE MODEL
num_correct = 0
num_samples = 0

model.load_state_dict(torch.load('best_model_weight.pth'))

model.eval()
with torch.no_grad():
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
