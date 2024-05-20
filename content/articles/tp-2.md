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

---

# Let's Get Started

Deep neural networks are powerful tools for tackling computer vision problems. They often outperform the simpler Multilayer Perceptrons (MLPs) that we've explored previously. In this tutorial, we'll dive into building Convolutional Neural Networks (CNNs or ConvNets).

By the end of this tutorial, you will know:

- **How Convolutional Neural Networks work**
- **Why CNNs are better than MLPs for image-related tasks**
- **How to create a CNN using PyTorch**
- **How to improve a CNN on a real use case**

## Why CNNs?

Convolutional Neural Networks (CNNs) are especially useful for working with images. Why? Because they are designed to recognize patterns, like edges, textures, and shapes, which are the building blocks of images.

One of the key features of CNNs is their **receptive field**. Think of it as a small window that scans over different parts of an image. Instead of looking at the whole picture at once, the CNN looks at small sections, one at a time, focusing on specific details. This helps the network detect local patterns and understand the image piece by piece.

Imagine trying to identify a cat in a photo. An MLP might struggle because it doesn't naturally focus on parts of the image, it'll try to look at everything at once. But a CNN starts by recognizing the edges of the cat, then the shapes like ears and eyes, and finally pieces these together to identify the whole cat. This "piece-by-piece" approach is why CNNs are so good at vision tasks and often outperform MLPs in these areas.

![cnn_filters](/images/tp-2/Visualization-of-example-features-of-eight-layers-of-a-deep-convolutional-neural.jpg)
*<center><small>Visualization of the weights/activations of several layers of a CNN which naturally form a hierarchy of filters that look like parts of the images it's analyzing! Source: Understanding Neural Networks Through Deep Visualization</small></center>*

In other words, CNNs have a built-in advantage: they are designed to tackle tasks where it's helpful to look at parts of an image (or a signal in general) and then combine those parts to understand the whole picture. This gives them a head start when learning the best weights during training because they are naturally better at processing image data than MLPs which have to figure out everything from scratch. In theory, you could train an MLP to behave like a CNN, but it would have to learn the convolution process first, putting it at a disadvantage in terms of learning speed. However, it works both ways, this bias towards convolution might not be the optimal solution for image tasks, and we're always looking for better methods. *Spoiler alert: Nowadays, we also use [transformers](https://arxiv.org/abs/2010.11929) and actually even [MLPs](https://arxiv.org/abs/2105.01601) for image tasks!*

### A CNN/ConvNet's Architecture

![cnn_arch](/images/tp-2/convnet_fig.png)

Let's take a look at the image above. Starting on the right, you'll see an _Outputs_ layer with two outputs. This means the network makes two types of predictions, which could be for two different classes in a classification task or two separate regression outputs.

To the left of this, we have two layers with hidden units, known as _Fully Connected_ layers. These are the same type of layers we know from a Multilayer Perceptron (MLP). So, even a Convolutional Neural Network (CNN) often includes MLP layers for making final predictions.

On the left side, we see **Convolution** layers followed by **[(Max) pooling](https://www.machinecurve.com/index.php/2020/01/30/what-are-max-pooling-average-pooling-global-max-pooling-and-global-average-pooling/)** layers.

So, what is a **convolution** operation?

In simple terms, a _convolution_ is a mathematical operation where one function modifies another. It's like combining two sets of information to create a new set. This process helps the network focus on important features in the images, like edges and textures.

Here's a more technical definition:

> In mathematics (in particular, functional analysis), convolution is a mathematical operation on two functions ($f$ and $g$) that produces a third function ($f ∗ g$).

*<center><small>Wikipedia</small></center>*

In our case, we will be mostly using 2D convolutions for images, but you can use 1D convolutions for other domains like text or audio data and even 3D convolutions for video data or MRI/CT scans.

![conv_op](/images/tp-2/conv_op.gif)
*<center><small>2D Convolution Block. Source: [Medium](https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1)</small></center>*

<notequote>
Note that while we say CNNs use convolution, they are actually using cross-correlation. It is a technicality, but in a CNN we do not flip the filter as is required in typical convolutions. However except for this flip, both operations are identical.
</notequote>

In other words, a Convolutional layer takes two parts and creates a function that shows how one changes the other. If you’re familiar with neural networks, you know they have inputs that pass through layers with weights. From a Convolution perspective, these layers also have weights and they measure how much the inputs "trigger" or affect these weights.

By adjusting the weights during training, we teach the network to recognize specific patterns in the input data. For example, a Convolutional layer can learn to be triggered by certain features, like a nose, and connect this to the output class "human."

![cnn_features](/images/tp-2/features_cnn.webp)
*<center><small>Hierarchy of convolution filters from lower layers (left) to higher layers (right). We ca sometimes visualize these filters since they should trigger when similar patterns are found in an image</small></center>*

Because CNNs use a sliding window (or kernel) that moves over the input data, they are **translation invariant**. This means they can detect features like a nose on a face regardless of where it appears in the image. This ability to recognize features anywhere in the image makes CNNs much more powerful for computer vision tasks than classic MLPs.

### Exploring the Internals of CNNs

It can be interesting to understand what these models are learning internally to know why convolution with filters is an interesting architectural choice for a neural network. It definitely helps in understanding why they work.

Let's take a look at what's called an [Activation Atlas](https://distill.pub/2019/activation-atlas/) below, specifically that of the [InceptionV1](https://arxiv.org/abs/1409.4842) CNN classifier (also known as "GoogLeNet") developed by Google that notably won the [2014 ImageNet Large Scale Visual Recognition Challenge](https://image-net.org/challenges/LSVRC/2014/index.php).

<questionquote>
Observe multiple layers and concepts in the interactive activation atlas below. What do you learn about CNN architectures?
</questionquote>

<iframe style="margin-left: -100px" width="800px" height="600px" src="https://distill.pub/2019/activation-atlas/app.html" title="activation_atlas"></iframe>

*<center><small>Credit: distill.pub</small></center>*

## Quick Dive into the Convolution Layer

As we already said above, the 2D convolution operation takes a filter (a fixed matrix of size $k * k$) and slides it across a signal (for example, an image). It is defined as follows:
 
$(f * g)(\mathbf{x}) = \int f(\mathbf{z}) g(\mathbf{x}-\mathbf{z}) d\mathbf{z}.$

However, the above definition is for continuous signals and we are working on a discrete signal (images contain pixels and there's nothing inbetween). Let's convert the above into a discrete formula:

$(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b).$

You'll note that there are two coordinates now, since we are working in two dimensions with image data.

<exercisequote>
Play with the below interactive convolution visualizer.
</exercisequote>

<iframe style="margin-left: -20px" width="600px" height="800px" src="https://ezyang.github.io/convolution-visualizer/" title="convolution_visualizer"></iframe> 

### Padding and Stride

To learn more about these and how to correctly compute your kernel size for a specific output, read this [article](https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html#padding-and-stride) which is quite well made.

## Let's build a CNN in PyTorch

```Python
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalizes it in the range [0.0, 1.0]
    download=True,                                  # Download the dataset if you don't already have it
)
```

---


Now that we have recalled how CNNs work, it's time to actually build one with PyTorch. Next, you will see a full example of a simple Convolutional Neural Network. From beginning to end, you will see that the following happens:

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

Now that we understand the basics, let's move to something more interesting.

<h2 style="display:inline;">Let's Aim Bigger: </h2>
<span style="margin-left:30px"><h1>Super-Resolution!</h1></span> 

### What is Super-Resolution?

Super resolution is the process of taking a low-resolution image and upscaling it to recover, or more accurately, extrapolate details at a higher resolution. The term "extrapolate" is used because the additional details do not actually exist in the original image. Unlike interpolation algorithms such as [bicubic](https://en.wikipedia.org/wiki/Bicubic_interpolation) or [Lanczos resampling](https://en.wikipedia.org/wiki/Lanczos_resampling), which do not require machine learning, super resolution uses advanced techniques to infer and generate these new details using neural networks.

Let's not forget that even these "simpler" interpolation methods are quite good baselines that can be hard to beat, even production-ready methods such as [AMD Radeon™ Super Resolution](https://www.amd.com/en/products/software/adrenalin/radeon-super-resolution.html) for video game super-resolution use Lanczos at their core. However, methods such as [Nvidia DLSS](https://www.nvidia.com/en-us/geforce/technologies/dlss/) use deep learning and offer better results and performance in general *(shameless plug since I worked on DLSS v1 :D)*.

![sr_example](/images/tp-2/sr_example.png)
*<center><small>Left: Low Resolution (LR) Image. Right: Reconstructed High Resolution (HR) Image</small></center>*

### History

The general idea is that we have a low resolution (LR) image and we want to get to a high resolution (HR) image. One of the first papers to do so using deep learning in an end-to-end fashion was "[Image Super-Resolution Using Deep Convolutional Networks, Chao Dong et al, 2014](https://arxiv.org/abs/1501.00092v3)".

![srcnn_fig_2](/images/tp-2/srcnn_fig_2.png)
*<center><small>Source: Image Super-Resolution Using Deep Convolutional Networks, Chao Dong et al, 2014</small></center>*

To enhance a low-resolution image, the image is first upscaled to the desired super-resolution size using bicubic interpolation, resulting in an interpolated image $Y$. The aim is to recover an image $F(Y)$ from $Y$ that resembles the high-resolution image $X$ we want to obtain in terms of dimensions. Despite $Y$ having the same dimensions as $X$, it is still referred to as low-resolution.

The process then involves three main operations: (1) Patch extraction and representation, where overlapping patches from $Y$ are extracted and represented as high-dimensional vectors to form feature maps. (2) Non-linear mapping, which transforms each high-dimensional vector into another high-dimensional vector representing high-resolution patches, forming a new set of feature maps. (3) Reconstruction, which combines these high-resolution patch representations to generate the final high-resolution image similar to $X$. These operations collectively form a convolutional neural network, as illustrated in the above figure.

Now, something that isn't ideal with their approach is that they first upscale the image using bicubic interpolation to get the final shape they need to operate on. This really isn't optimal at all since it introduces a bias from the bicubic interpolation!

### Let's Improve The Architecture

Let's improve their architecture step-by-step by going for an architecture that takes the low resolution image directly as input.

![model_svg|100x50](/images/tp-2/model.gv.svg)

<exercisequote>
Study and recreate the above architecture in PyTorch.
</exercisequote>

A few hints:
  - The input dimensions of the network $(1, 1, H, W)$ are $(batch\\_size, channels, height, width)$
  - Why is $channels = 1$?
  - What's the `PixelShuffle` operation?
  - Is padding needed in the convolutions? Read more about this [here](https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html#padding-and-stride).

#### RGB for Images, right? Nah, Let's Switch To YCbCr!

Computer displays are driven by red ($R$), green ($G$), and blue ($B$) voltage signals. These signals are efficient in describing how color physically works. However, these RGB signals are not efficient to store since they have a lot of redundancy and usually we want to have the best compromise between size and quality.

On top of that, and that's very important for us here, the human vision works in a specific way which was can use to our advantage: Due to the densities of color and brightness-sensitive receptors in the human eye, humans can see considerably more fine detail in the brightness of an image (the $Y$ component) than in the hue and color saturation of an image (the $Cb$ and $Cr$ components).

This property enables us to decorrelate "detail" from "color" information in YCbCr while RGB has luminance information spread out across $R$, $G$ and $B$. And as you know, the less correlation we have between features (or components in this case) in data, the more we can compress this data. JPEG actually uses YCbCr since it can help bette compress the final image.

*Now, why would we need this inside a neural network?*

Because for our first simple super resolution network, which just adds more details to an image, we'll only apply super-resolution on the "details" or $Y$ channel and it'll be enough to recombine it with bicubic upscaled $Cb$ and $Cr$ channels which we'll convert back to RGB so we can see the final image.

This basically explains why $channels = 1$ in our network.

#### What's Pixel Shuffle?

If you look at our network architecture above, we want to get from an image input size of $(H, W)$ to an image output size of $(H * 2, W * 2)$ to achieve a super-resolution scaling of $2×$. To achieve this, there are several solutions that exist, of which are:
   - Interpolation methods: This simply applies an interpolation algorithm such as bicubic interpolation and gives us the correct output size, but at the price of blurry features.
   - Transposed convolution: This is a different type of convolution that outputs a bigger signal by effectively spacing the signal before applying convolution to it, so it makes it bigger in the output, but it's also a form of interpolation in the end that discards part of the information. More details [here](https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11).

<center><figure class="half" style="display:flex">
    <img style="width:400px; margin-right: 10px; margin-left: -100px" src="/images/tp-2/interpolation_sr.png">
    <img style="width:400px" src="/images/tp-2/strided_sr.png">
</figure></center>

*<center><small>Left: Forming high res feature maps using interpolation. Right: Forming high res feature maps using transposed convolution</small></center>*

Pixel Shuffle was introduced to circumvent all the above issues by rearranging the input tensors from depth to spatial features. This means that as we go deeper in our network, we'll create enough convolution filters (this is the depth part) that will in the end create feature maps ordered by depth to be used to store the new details we will add to our image by rearranging them from depth to space.

This means that our network will do what any usual CNN does by stacking convolutional layers with a lot of filters to encode the input image into hidden feature maps and then we'll reduce these filters until the end of the network until we have only enough left for storing the new image details.

So at the end of the network in our case, we'll have a final convolution layer that will store enough filters to propagate them into the spatial dimensions of our model. Let's trace the shape of the image feature maps through our network:
  - Input luminance (black & white) image size is $(1, 1, H, W)$
  - Classic convolutional process with $ReLU(conv2d(X))$ operations
  - At the end of convolutional layers, we downscale filters to exactly $(1, 4, H, W)$
    - Why? Because scaling an image $2×$ means scaling each of its dimensions (width and height) by $2×$, which is a $4×$ factor in the end
  - Pixel Shuffle propagates the $4$ filters into $2×$ on the width and $2×$ on the height dimensions: $(1, 1, H × 2, W × 2)$

![pixel_shuffle](/images/tp-2/pixel_shuffle.gif)

*<center><small>Pixel Shuffle propagates depth information ($z$ axis) to spatial information ($x$ axis and $y$ axis)</small></center>*

From this, we can conclude that the Pixel Shuffle formula works for any resolution upscaling factor: It just rearranges elements from a tensor of shape $(∗,C×r^2,H,W)$ to a tensor of shape $(∗,C,H×r,W×r)$, where $r$ is an upscale factor.

PyTorch fortunately defines [Pixel Shuffle](https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html) as a very simple layer for us that we can just plug into our network.

<exercisequote>
Plug the network architecture you made above into the code repository that trains a super-resolution model.
</exercisequote>

You should get a network that'll create higher resolution image such as this:

<center><figure class="half" style="display:flex">
    <img style="width:400px; margin-right: 10px; margin-left: -100px" src="/images/tp-2/orig_low_res.jpg">
    <img style="width:400px" src="/images/tp-2/new_high_res.png">
</figure></center>

*<center><small>Left: Low resolution image we give our CNN. Right: The output of the SuperRes CNN</small></center>*

<exercisequote>
Using the `einops` library which uses an Einstein-inspired notation, rewrite Pixel Shuffle from scratch using the `rearrange` operation. Then retrain the model and check that you have the exact same results as before.
</exercisequote>

---

# You're Done!

Great job, you made it this far!

Send it to my [email adress](mailto:chady1.dimachkie@epita.fr?subject=TP%202) with the subject **Practical Work 2**: [chady1.dimachkie@epita.fr](mailto:chady1.dimachkie@epita.fr?subject=Practical%20Work%202)

**Don't hesitate if you have any questions!**

→ [Coming Next: Practical Work 3](/articles/)
