# Deep-Learning-on-CIFAR100
This code was created for a Computer Vision class. All code was run on Google Colab with the GPU enabled.

Per the project requirements, the code is divided into two parts.

Part 1:
- Build a CNN to classify images from the dataset CIFAR100 using VGG16 with pretrained weights from ImageNet.

Steps taken:
1. Import modules and libraries
2. Create the training and testing sets, and their respective data loaders
3. Prepare model with VGG16 pre-trained
4. Extract the number of input features for the last fully connected layer of the model
5. Replace the last fully connected layer with a new layer.
6. Set up the hyperparameters
7. Train the model
8. Test the model and report the accuracy as a percent

**Part 2**
- Build your own CNN to perform classification on CIFAR100.
CNN Requirements:
- Must have 3 convolutional layers
- Must have 2 fully connected layers
- Must use max-pooling layers
- Activation function must be ReLU

Steps taken:

