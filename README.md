# Deep-Learning-on-CIFAR100
This code was created for a Computer Vision class. All code was run on Google Colab with the GPU enabled.

Per the project requirements, the code is divided into two parts, but the full Jupyter Notebook is included in the
repository for clarity.

**Part 1:**
- Build a CNN to classify images from the dataset CIFAR100 using VGG16 with pretrained weights from ImageNet.

Steps taken:
1. Import modules and libraries
2. Create the training and testing sets, and their respective data loaders
3. Prepare model with VGG16 pre-trained
4. Extract the number of input features for the last fully connected layer of the model
5. Replace the last fully connected layer with a new layer.
6. Set up the hyperparameters
7. Train the model
8. Test the model
9. Report the accuracy as a percent

**Part 2**
- Build your own CNN to perform classification on CIFAR100.
CNN Requirements:
- Must have 3 convolutional layers
- Must have 2 fully connected layers
- Must use max-pooling layers
- Activation function must be ReLU

Steps taken:
1. Assume the modules and libraries are imported and the training, testing, and data loaders are created
2. Create our own CNN by first defining the init function. Here we create the NN based on the project specifications
3. Write the forward pass function. Here, overwrite the built-in forward function so PyTorch uses our new function.
4. Set up the hyperparameters
5. Train the model
6. Test the model
7. Report the accuracy as a percent
