---
title: 1. Build a Neural Network in PyTorch
date: 2024-03-06
images: 
- https://cs231n.github.io/assets/nn1/neural_net.jpeg
---

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript"
  src="http://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

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

What's a `simple linear looking function you ask`?

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



# Send it to me!

Send it to my [email adress](mailto:chady1.dimachkie@epita.fr?subject=TP%201) with the subject **Practical Work 1**: [chady1.dimachkie@epita.fr](mailto:chady1.dimachkie@epita.fr?subject=TP%201)

→ [TP 2](/articles/structure)
