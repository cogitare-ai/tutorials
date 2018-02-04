"""
Training a Convolutional Neural Network for Image Classification
================================================================

*Tutorial adapted from http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html*

Generally, when you have to deal with image, text, audio or video data,
you can use standard python packages that load data into a numpy array.
Then you can convert this array into a ``torch.*Tensor``.

-  For images, packages such as Pillow, OpenCV are useful.
-  For audio, packages such as scipy and librosa
-  For text, either raw Python or Cython based loading, or NLTK and
   SpaCy are useful.

For this tutorial, we will use the CIFAR10 dataset. It has the classes:
``airplane``, ``automobile``, ``bird``, ``cat``, ``deer``, ``dog``,
``frog``, ``horse``, ``ship``, ``truck``. The images in CIFAR-10 are of
size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

.. figure:: /_static/cifar10.png
   :alt: cifar

   cifar

Training an image classifier
----------------------------

We will do the following steps in order:

1. Import libraries and add model settings
2. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
3. Define a Convolution Neural Network (forward)
4. Train the network on the training data
5. Test the network on the test data

"""


######################################################################
# 1. Importing and Declaring Global Variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# We start be importing all libraries that will be required in this
# tutorial.
# 

from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from cogitare.data import DataSet, AsyncDataLoader
from cogitare import utils, Model
from cogitare.plugins import EarlyStopping
from cogitare.metrics.classification import accuracy
import cogitare

import argparse

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
pa = parser.add_argument  # define a shortcut

pa('--batch-size', help='Size of the training batch', type=int, default=64)
pa('--cuda', help='enable cuda', action='store_true')
pa('--dropout', help='dropout rate in the input data', type=float, default=0.3)
pa('--learning-rate', help='learning rate', type=float, default=0.001)
pa('--max-epochs', help='limit the number of epochs in training', type=int, default=10)


# load the model arguments
try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


cogitare.utils.set_cuda(args.cuda)

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


######################################################################
# 2. Loading and normalizing CIFAR10
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Using ``torchvision``, it’s extremely easy to load CIFAR10.
# 

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load the CIFAR 10 data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                       download=True, transform=transform)

print(type(trainset.train_data))


######################################################################
# Torchvision loads the data as numpy arrays.
# 
# We now create two datasets to hold the train and test sets.
# 

print(type(trainset.train_data))

print(type(trainset.train_labels))

def batch2variable(batch):
    data, label = batch
    data = utils.to_tensor(data)
    
    # B x W x H x C to B x C x W x W
    data = data.transpose(1, 2).transpose(1, 3)
    
    return utils.to_variable(data, dtype=torch.FloatTensor), utils.to_variable(label)

# convert the trainset.train_labels to LongTensor, instead of python list

data_train = DataSet([trainset.train_data, torch.LongTensor(trainset.train_labels)],
                     batch_size=args.batch_size,
                     drop_last=True)

# use the async loader, to pre-load 8 batches ahead of the model
# each batch is then loaded and moved to a torch Variable
data_train = AsyncDataLoader(data_train, buffer_size=args.batch_size * 8,
                             on_batch_loaded=batch2variable)

data_test = DataSet([testset.test_data, torch.LongTensor(testset.test_labels)],
                    batch_size=args.batch_size,
                    drop_last=True)
data_test = AsyncDataLoader(data_test, buffer_size=args.batch_size * 8,
                            on_batch_loaded=batch2variable)

# fill the data buffer
data_train.cache()
data_test.cache()


######################################################################
# The train and test datasets are defined as a collection of tuples, each
# tuple contains ``(data, expected label)``.
# 

print(next(data_train))


######################################################################
# 2.1 Data Visualization
# ----------------------
# 
# Let us show some of the training images, for fun.
# 

def imshow(img):
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def showlabels(labels, qtd):
    for j in range(1, qtd + 1):
        print('%10s' % CLASSES[int(labels[j - 1])], end='')

        if j % 4 == 0:
            print('\n')

images, labels = next(data_train)
print(images.shape)

imshow(torchvision.utils.make_grid(images.data[:16], nrow=4))
showlabels(labels, 16)

######################################################################
# 3. Define a Convolution Neural Network
# --------------------------------------
# 
# In this section, we’ll define the forward method of the Cogitare Model.
# In Cogitare, you must implement two methods in the model: **forward**
# and **loss**.
# 
# This is a Convolutional Neural Network (CNN) for Image Classification.
# 

class CNN(Model):
    
    def __init__(self):
        super(CNN, self).__init__()
            
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, batch):
        # in this sample, each batch will be a tuple
        # containing (input_batch, expected_batch)
        # in forward in are only interested in input so that we 
        # can ignore the second item of the tuple
        x, _ = batch      
        
        x = F.dropout(x, args.dropout)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)
    
    def loss(self, output, batch):
        # in this sample, each batch will be a tuple 
        # containing (input_batch, expected_batch)
        # in loss in are only interested in expected so that
        # we can ignore the first item of the tuple
        _, expected = batch

        return F.nll_loss(output, expected)


######################################################################
# The model class is simple; it only requires de forward and loss methods.
# By default, Cogitare will backward the loss returned by the loss()
# method, and optimize the model parameters.
# 


######################################################################
# 4. Training the Model
# ---------------------
# 
# We first define the model optimizer for training the model.
# 

cnn = CNN()

optimizer = optim.Adam(cnn.parameters(), lr=args.learning_rate)


######################################################################
# We now add the default plugins to watch the training status. The default
# plugin includes:
# 
# -  Progress bar per batch and epoch
# -  Plot training and validation losses (if validation_dataset is
#    present)
# -  Log training loss
# 
# And some extra plugins.
# 

cnn.register_default_plugins()

early = EarlyStopping(max_tries=5, path='/tmp/model.pt')
# after 5 epochs without decreasing the loss, stop the 
# training and the best model is saved at /tmp/model.pt

# the plugin will execute in the end of each epoch
cnn.register_plugin(early, 'on_end_epoch')


######################################################################
# We can now run the training:
# 

if args.cuda:
    cnn = cnn.cuda()

cnn.learn(data_train, optimizer, data_test, max_epochs=args.max_epochs)


######################################################################
# 5. Model Evaluation
# -------------------
# 
# We now check the model loss and accuracy on the test set:
# 

def model_accuracy(output, data):
    _, indices = torch.max(output, 1)

    return accuracy(indices, data[1])

# evaluate the model loss and accuracy over the validation dataset
metrics = cnn.evaluate_with_metrics(data_test, {'loss': cnn.metric_loss, 'accuracy': model_accuracy})

# the metrics is an dict mapping the metric name (loss or accuracy, in this sample) to a list of the accuracy output
# we have a measurement per batch. So, to have a value of the full dataset, we take the mean value:

metrics_mean = {'loss': 0, 'accuracy': 0}
for loss, acc in zip(metrics['loss'], metrics['accuracy']):
    metrics_mean['loss'] += loss
    metrics_mean['accuracy'] += acc.data[0]

qtd = len(metrics['loss'])

print('Loss: {}'.format(metrics_mean['loss'] / qtd))
print('Accuracy: {}'.format(metrics_mean['accuracy'] / qtd))


######################################################################
# 5.1 Visualization
# ~~~~~~~~~~~~~~~~~
# 
# To check how the model behaves, we can plot the images, the expected and
# predicted labels
# 

images, labels = next(data_test)

imshow(torchvision.utils.make_grid(images.data[:16], nrow=4))


######################################################################
# We forward the data to get the model output to the batch above.
# 

predicted = cnn.predict((images, None))
# remember that forward method expect a tuple, where the 
# first item contains the data to be passed in the net
predicted.shape

_, predicted_labels = torch.max(predicted, dim=1)
print('Predicted:\n')
showlabels(predicted_labels[:16], 16)
