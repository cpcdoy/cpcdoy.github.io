---
title: 1. Build a Neural Network in PyTorch
date: 2024-03-06
images: 
- https://cs231n.github.io/assets/nn1/neural_net.jpeg
---


In this first practical work, we'll be experimenting with simple neural networks on simple datasets.

# Get Setup

It's recommended to use a Jupyter notebook since you'll be doing some visualization.

## Locally

You should have `Jupyter` installed and the following dependencies. You can use `pip` or any other dependency manager:

```sh
pip install torch matplotlib
```

## In The Cloud *(Google Colab)*

Simply click [here to open a Colab notebook](https://colab.research.google.com/) in your browser. You'll need to sign-in with you Google account.

# Let's Get Started

In this first practical work you'll be creating a simple neural network in PyTorch to fit multiple simple datasets.

**Let's get started !**

## Why PyTorch?

![pytorch](https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png)

PyTorch is one of the leading open-source machine learning frameworks to design, train and run inference on neural networks. It is developped by Meta AI.

A lot of machine learning software made by big tech companies is made using PyTorch, even Tesla for their Autopilot.

## Our First Neural Network!

### Let's start with a linear function

Before starting and creating a neural network, we need to choose a dataset to use so we can experiment on.

To understand the basics when building a neural network, we'll first create our own dummy dataset consisting of a simple linear looking function.

What's a *simple linear looking function* you ask?

Something of the form $y = ax + b$, let's say for example $y = -x + 5$, where $y$ will be our dataset that contains all the possible points that will be created by all the possible $x$ values we input.

![linear_line](https://upload.wikimedia.org/wikipedia/commons/thumb/0/0e/Linear_Function_Graph.svg/1280px-Linear_Function_Graph.svg.png)
*<center><small>Example linear functions.</small></center>*

Let's recreate this function in PyTorch:

```Python
def create_dataset(n_data:int = 100, a:float = -1.0, b:float = 5, scale:int = 15) -> torch.tensor:
    # Generate random data as inputs
    x = torch.rand(n_data, 1) * scale  # n_data=100 random numbers scaled between 0 and scale

    # Calculate corresponding y values
    y = a * x + b

    # We center the data by using the mean of all values
    return x - torch.mean(x), y - torch.mean(y)
```

![linear_fit](/tp_1_linear_fit_no_noise.png)

Now, let's make it slightly more interesting and make it only look like it's a linear function. For that we'll add some noise. You'll see later that data you find in real life has a lot of noise, meaning it's not perfect.


```Python
def create_dataset(n_data:int = 100, a:float = -1.0, b:float = 5, scale:int = 15) -> torch.tensor:
    # Generate random data as inputs
    x = torch.rand(n_data, 1) * scale  # n_data=100 random numbers scaled between 0 and scale

    # Calculate corresponding y values
    y = a * x + b

    # Add some noise
    noise = torch.randn(y.shape) * 2  # Gaussian noise with standard deviation of 2
    y = y + noise

    # We center the data by using the mean of all values
    return x - torch.mean(x), y - torch.mean(y)
```

![linear_fit](/tp_1_linear_fit.png)

The data looks linear if you look at it from far away which is what we wanted. Let's try to make a simple neural network that will learn to approximately recreate this function.

### Creating a Linear Neural Network

Let's first start by describing what a linear neural network looks like:

$y = W^ta + b_w$ where $W$ is the weights matrix and $b$ is the bias. Unsurprisingly, it reminds us of the above previous linear equation $y = ax + b$.

From this equation, we can see that if we're able to learn a $W$ that is close to $a$ and a $b_w$ close to $b$ then we'll have approximated our function.

```Python
class Net(torch.nn.Module):
    """
    A simple 1 layer neural network with no activations
    """

    def __init__(self, n_feature:int, n_output:int):
        super(Net, self).__init__()
        
        # Create a linear layer
        self.hidden = torch.nn.Linear(n_feature, n_output)

    def forward(self, x:torch.tensor):
        # Use the linear layer
        x = self.hidden(x)

        return x
```

The above class is very simple, it simply creates one linear layer that will be our entire neural network. You'll see that this `Linear` layer is the basis of most neural networks, even the most complex ones.

We have two functions, the `__init__` function just initializes the network. Basically, we want our $W$ and $b_w$ values to have an initial value and PyTorch automatically initializes our variables in a way that will make the network converge quickly. You can read more about initialization [here](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_), but we'll come back to it later.

Let's instanciate the neural network:

```Python
# We define the network
net = Net(n_feature=1, n_output=1)
```

As simple as that!

The network only takes one input and outputs a single number since we want the network to map each $x$ to its corresponding $y$ value by doing the $ax + b$ transformation.

<questionquote>
What is a &nbsp<i>Linear Neural Network</i>&nbsp equivalent to?
</questionquote>

### Training the network

#### The Theory

Training a neural network means that we want to find a path towards one of the best solutions to our problem. The *solution* should be the best set of weights we can find (when our model converges and has the desired accuracy) from our *starting point* (the initialization of the network). How do we go from a *starting point* to a *solution*, we need to use an algorithm that helps us navigate there.

One of these algorithms is called *gradient descent*: *gradient* because we use derivatives and *descent* because we use these *derivaties* to guide us towards a good solution in the *loss landscape*.

##### What's the *loss function*?

The *loss function* is a function that helps us compute a distance between of how far or close are our model's prediction compared to what it's supposed to answer (the *ground truth*). The *loss function* needs to be *differentiable* everywhere.

A simple and common loss function is the [*Mean Squared Error*](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) loss (also called *squared L2 norm*): $\frac{1}{n}\sum{(y_i - \hat{y}_i)^2}$, where $y_i$ is the ground truth and $\hat{y}_i$ is the predicted value from our model.

<notequote>
Why not just use the difference without squaring? Because the squaring actually penalizes mistakes and outliers more!
</notequote>

There are many more [loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions) and we'll see a few others along the way.

##### Ok, so what's the *loss landscape*?

Take a look at this mountain looking picture:

![loss_landscape](/loss_landscape.jpeg)
*<center><small>Example of a loss landscape.</small></center>*

This is basically a representation of the loss values around the weight space of the network: What this means is that for each possible weights our networks can have (it's a very large number so we usually only ever explore a few like what you see on the image), we can plot how it performs thanks to the loss function.

The <span style="color: blue">lower</span> we are in the landscape, the closer we get to a good solution. Furthermore, you can see that we start from a **point** quite <span style="color: red">high</span> in the lanscape (the initialization of network which we hope is good enough) and then we <span style="color: green">explore by following the gradients</span> of what the loss is telling us until we reach a <span style="color: blue">good enough weight set</span> where our model performs well.

<notequote>
It's very probable that we'll miss the most optimal solution when navigating in the loss landscape. In practice, we'll find there are many quasi-optimal solutions that exist that are sufficient.
</notequote>

##### How do we navigate this landscape?

To navigate this landscape we'll use what's called an *optimizer*. The optimizer will follow what the derivates of the loss function at a specific time step $t$ is telling us and will take the best guess as to where to go given this information.

Most neural network optimizers are based on *Gradient Descen (GD)*. This algorithm looks like this:

$W_{i+1} = W_i - lr * D_{loss}(W_i, X_{train}, y_{train})$, where $W_i$ are the current weights of the model, $D_{loss}$ is the loss function's derivative that takes as input the current model weights $W_i$ to compute predictions on the training set $X_{train}$ and compare the model output (which we called $\hat{y}$ earlier) to the ground truth labels $y$.

We iterate on this until we find a model that performs well on our test data, which should be correlated to a <span style="color: blue">low loss</span> value.

#### In Practice

In practice, PyTorch does everything for you and defines preexisting algorithms we can reuse.

Above we said that we needed two things:
- An algorithm to find the best path to a good set of weights: This is called an *optimizer* and PyTorch offers a lot of different ones in [torch.optim](https://pytorch.org/docs/stable/optim.html).
 - A *loss function* to define how well our model is performing at each time step: [torch.nn.*Loss](https://pytorch.org/docs/stable/nn.html#loss-functions) offers a lot of loss functions.

In our case, we'll use:
- Optimizer: [Stochastic Gradient Descent (SGD)](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
- Loss function: [Mean Squared Error (MSE)](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)

<questionquote>
What is the difference between Stochastic Gradient Descent (SGD) and the usual Gradient Descent (GD)?
</questionquote>

```Python
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
loss_func = torch.nn.MSELoss()
```



# Send it to me!

Send it to my [email adress](mailto:chady1.dimachkie@epita.fr?subject=TP%201) with the subject **Practical Work 1**: [chady1.dimachkie@epita.fr](mailto:chady1.dimachkie@epita.fr?subject=Practical%20Work%201)

→ [TP 2](/articles/structure)
