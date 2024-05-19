---
title: 2. Intro to Convolutional Neural Networks (CNNs) in PyTorch
date: 2024-05-19
images: 
- /images/tp-2/cnn_tp_cover.webp
---

In this second practical work, we'll be experimenting with Convolutional Neural Networks (CNNs).

# As Usual, Get Setup

It's recommended to use a Jupyter notebook since you'll be doing some visualization.

## Locally

You should have `Jupyter` installed and the following dependencies. You can use `pip` or any other dependency manager:

```sh
pip install torch torchvision matplotlib scikit-learn
```

You can add more dependencies if you think it is relevant.

## In The Cloud *(Google Colab)*

Simply click [here to open a Colab notebook](https://colab.research.google.com/) in your browser. You'll need to sign-in with you Google account.

In the first cell, add this following to install the dependencies:

```sh
!pip install torch torchvision matplotlib scikit-learn
```

Same here, you can add more dependencies if you think it is relevant.

# Let's Get Started

Deep neural networks are powerful tools for tackling computer vision problems. They often outperform the simpler Multilayer Perceptrons (MLPs) that we've explored previously. In this tutorial, we'll dive into building Convolutional Neural Networks (ConvNets).

By the end of this tutorial, you will know:

- **How Convolutional Neural Networks (CNNs or ConvNets) work**
- **Why CNNs are better than MLPs for image-related tasks**
- **How to create a CNN using PyTorch**

## Why CNNs?

Convolutional Neural Networks (CNNs) are especially useful for working with images. Why? Because they are designed to recognize patterns, like edges, textures, and shapes, which are the building blocks of images.

One of the key features of CNNs is their **receptive field**. Think of it as a small window that scans over different parts of an image. Instead of looking at the whole picture at once, the CNN looks at small sections, one at a time, focusing on specific details. This helps the network detect local patterns and understand the image piece by piece.

Imagine trying to identify a cat in a photo. An MLP might struggle because it doesn't naturally focus on parts of the image, it'll try to look at everything at once. But a CNN starts by recognizing the edges of the cat, then the shapes like ears and eyes, and finally pieces these together to identify the whole cat. This "piece-by-piece" approach is why CNNs are so good at vision tasks and often outperform MLPs in these areas.

![cnn_filters](/images/tp-2/Visualization-of-example-features-of-eight-layers-of-a-deep-convolutional-neural.jpg)
*<center><small>Visualization of the weights/activations of several layers of a CNN which naturally form a hierarchy of filters that look like parts of the images it's analyzing! Source: Understanding Neural Networks Through Deep Visualization</small></center>*

In other words, CNNs have a built-in advantage: they are designed to tackle tasks where it's helpful to look at parts of an image (or a signal in general) and then combine those parts to understand the whole picture. This gives them a head start when learning the best weights during training because they are naturally better at processing image data than MLPs, which have to figure out everything from scratch. In theory, you could train an MLP to behave like a CNN, but it would have to learn the convolution process first, putting it at a disadvantage in terms of learning speed. However, it works both ways, this bias towards convolution might not be the optimal solution for image tasks, and we're always looking for better methods. *Spoiler alert: Nowadays, we also use transformers for image tasks!*

### A CNN/ConvNet's Architecture

![cnn_arch](/images/tp-2/convnet_fig.png)

Let's take a look at the image above. Starting on the right, you'll see an _Outputs_ layer with two outputs. This means the network makes two types of predictions, which could be for two different classes in a classification task or two separate regression outputs.

To the left of this, we have two layers with hidden units, known as _Fully Connected_ layers. These are the same type of layers we know from a Multilayer Perceptron (MLP). So, even a Convolutional Neural Network (CNN) often includes MLP layers for making final predictions. But what makes this network _Convolutional_?

It's the Convolutional layers (obviously).

On the left side, we see **Convolution** layers followed by **[(Max) pooling](https://www.machinecurve.com/index.php/2020/01/30/what-are-max-pooling-average-pooling-global-max-pooling-and-global-average-pooling/)** layers.

So, what is a **convolution** operation?

In simple terms, a _convolution_ is a mathematical operation where one function modifies another. It's like combining two sets of information to create a new set. This process helps the network focus on important features in the images, like edges and textures.

Here's a more technical definition:

> In mathematics (in particular, functional analysis), convolution is a mathematical operation on two functions ($f$ and $g$) that produces a third function ($f ∗ g$).

*<center><small>Wikipedia</small></center>*

<notequote>
Note that while we say CNNs use convolution, they are actually using cross-correlation. It is a technicality, but in a CNN we do not flip the filter as is required in typical convolutions. However except for this flip, both operations are identical.
</notequote>

In other words, a Convolutional layer takes two parts and creates a function that shows how one changes the other. If you’re familiar with neural networks, you know they have inputs that pass through layers with weights. From a Convolution perspective, these layers also have weights and they measure how much the inputs "trigger" or affect these weights.

By adjusting the weights during training, we teach the network to recognize specific patterns in the input data. For example, a Convolutional layer can learn to be triggered by certain features, like a nose, and connect this to the output class "human."

Because ConvNets use a sliding window (or kernel) that moves over the input data, they are **translation invariant**. This means they can detect features like a nose regardless of where it appears in the image. This ability to recognize features anywhere in the image makes ConvNets much more powerful for computer vision tasks than classic MLPs.

## Code example: simple Convolutional Neural Network with PyTorch

Now that we have recalled how ConvNets work, it's time to actually build one with PyTorch. Next, you will see a full example of a simple Convolutional Neural Network. From beginning to end, you will see that the following happens:

1. **The imports**. First of all, we're importing all the dependencies that are necessary for this example. For loading the dataset, which is `MNIST`, we'll need the operating system functionalities provided by Python - i.e., `os`. We'll also need PyTorch (`torch`) and its neural networks library (`nn`). Using the `DataLoader` we can load the dataset, which we can transform into Tensor format with `transforms` - as we will see later.
2. **The neural network Module definition.** In Pytorch, neural networks are constructed as `nn.Module` instances - or neural network modules. In this case, we specify a `class` called `ConvNet`, which extends the `nn.Module` class. In its constructor, we pass some data to the super class, and define a `Sequential` set of layers. This set of layers means that a variety of neural network layers is stacked on top of each other.
3. **The layers**. Recall from the image above that the first layers are Convolutional in nature, followed by MLP layers. For two-dimensional inputs, such as images, Convolutional layers are represented in PyTorch as `nn.Conv2d`. Recall that all layers require an activation function, and in this case we use Rectified Linear Unit (`ReLU`). The multidimensional output of the final Conv layer is flattened into one-dimensional inputs for the MLP layers, which are represented by `Linear` layers.
4. **Layer inputs and outputs.** All Python layers represent the number of _in\_channels_ and the number of _out\_channels_ in their first two arguments, if applicable. For our example, this means that:
    - The first `Conv2d` layer has one input channel (which makes sence, since MNIST data is grayscale and hence has one input channel) and provides ten output channels.
    - The second `Conv2d` layer takes these ten output channels and outputs five.
    - As the MNIST dataset has 28 x 28 pixel images, two `Conv2d` layers with a kernel size of 3 produce feature maps of 24 x 24 pixels each. This is why after flattening, our number of inputs will be `24 * 24 * 5` - 24 x 24 pixels with 5 channels from the Conv layer. 64 outputs are specified.
    - The next Linear layer has 64 inputs and 32 outputs.
    - Finally, the 32 inputs are converted into 10 outputs. This also makes sence, since MNIST has ten classes (the numbers 0 to 9). Our loss function will be able to handle this format.
5. **Forward definition**. In the `forward` def, the forward pass of the data through the network is performed.
6. **The operational aspects**. Under the `main` check, the random seed is fixed, the data is loaded and preprocessed, the ConvNet, loss function and optimizer are initialized and the training loop is performed. In the training loop, batches of data are passed through the network, after the loss is computed and the error is backpropagated, after which the network weights are adapted during optimization.

```python
import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

class ConvNet(nn.Module):
  '''
    Simple Convolutional Neural Network
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(1, 10, kernel_size=3),
      nn.ReLU(),
      nn.Conv2d(10, 5, kernel_size=3),
      nn.ReLU(),
      nn.Flatten(),
      nn.Linear(24 * 24 * 5, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 10)
    )


  def forward(self, x):`
    '''Forward pass'''
    return self.layers(x)
  
  
if __name__ == '__main__':
  
  # Set fixed random number seed
  torch.manual_seed(42)
  
  # Prepare CIFAR-10 dataset
  dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
  trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
  
  # Initialize the ConvNet
  convnet = ConvNet()
  
  # Define the loss function and optimizer
  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(convnet.parameters(), lr=1e-4)
  
  # Run the training loop
  for epoch in range(0, 5): # 5 epochs at maximum
    
    # Print epoch
    print(f'Starting epoch {epoch+1}')
    
    # Set current loss value
    current_loss = 0.0
    
    # Iterate over the DataLoader for training data
    for i, data in enumerate(trainloader, 0):
      
      # Get inputs
      inputs, targets = data
      
      # Zero the gradients
      optimizer.zero_grad()
      
      # Perform forward pass
      outputs = convnet(inputs)
      
      # Compute loss
      loss = loss_function(outputs, targets)
      
      # Perform backward pass
      loss.backward()
      
      # Perform optimization
      optimizer.step()
      
      # Print statistics
      current_loss += loss.item()
      if i % 500 == 499:
          print('Loss after mini-batch %5d: %.3f' %
                (i + 1, current_loss / 500))
          current_loss = 0.0

  # Process is complete.
  print('Training process has finished.')
```

# Let's Do Something Useful: Super-Resolution!

## What is super-resolution?

## Let's Make a Custom Architecture

### Pixel-Shuffle Operation

### 

# You're Done!

Great job, you made it this far!

Send it to my [email adress](mailto:chady1.dimachkie@epita.fr?subject=TP%202) with the subject **Practical Work 2**: [chady1.dimachkie@epita.fr](mailto:chady1.dimachkie@epita.fr?subject=Practical%20Work%202)

**Don't hesitate if you have any questions!**

→ [Coming Next: Practical Work 3](/articles/)
