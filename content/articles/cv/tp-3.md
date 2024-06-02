---
title: 3. Intro to Denoising Diffusion Probabilistic Models (DDPMs) for Image Generation
date: 2024-06-02
images:
- /images/tp-3/diffusion_model_arch.png
---

In this third practical work, we'll be experimenting with **Denoising Diffusion Probabilistic Models (DDPMs)**.

# As Usual, Get Setup

## Locally

You should have `Jupyter` installed and the following dependencies. You can use `pip` or any other dependency manager:

```sh
pip install torch einops datasets matplotlib tqdm
```

You can add more dependencies if you think it is relevant.

## In The Cloud *(Google Colab)*

Simply [click here to open a Colab notebook](https://colab.research.google.com/) in your browser. You'll need to sign-in with you Google account.

In the first cell, add the following to install the dependencies:

```sh
!pip install torch einops datasets matplotlib tqdm
```

Same here, you can add more dependencies if you think it is relevant.

---

# Let's Get Started

You've probably already heard of [DALL-E](https://openai.com/index/dall-e-3/), [Midjourney](https://www.midjourney.com/home) or even [Stable Diffusion](https://stability.ai/news/stable-diffusion-3) where you give them a piece of text with instruction (prompt) and they generate any image you can ask for.

![dall_e_3](/images/tp-3/dall_e_3.webp)
*<center><small>DALL-E 3 Prompt: Imagine a vibrant and picturesque street in La Perla, Puerto Rico, styled in a digital art medium. Colorful houses line the way, with lush, tropical flora accentuating the architecture. Umbrellas add pops of color along the street, which leads to a serene beach where gentle waves meet the shore under a sunset sky.</small></center>*

**Denoising Diffusion Probabilistic Models (DDPMs)**, also called Diffusion Models, work really well and they've beaten other long standing architectures like Generative Adversarial Networks (GANs) at image generation tasks.

## Denoising Diffusion Probabilistic Models (DDPMs)

In this section, we'll be exploring the original "Denoising Diffusion Probabilistic Models" paper by [Ho et al, 2020](https://arxiv.org/abs/2006.11239).

### The Main Idea

These models have already existed for a while, but it took quite some iterations to make them work as well as today. It first started off from ideas from "non-equilibrium statistical physics" which sounds quite complex but is essentially the science of evolution of isolated systems over time according to deterministic equations. Examples of these systems include:
- Heat distribution inside of a material
- Electric currents carried by the motion of charges in a conductor
- Chemical Reactions

*Ok, but what does this have to do with Machine Learning?*

Well, this method was first introduced in the paper titled "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" by [Sohl-Dickstein et al, 2015](https://arxiv.org/abs/1503.03585). In their paper, they propose the following method:

> The essential idea, inspired by non-equilibrium statistical physics, is to systematically and slowly destroy structure in a data distribution through an iterative forward diffusion process. We then learn a reverse diffusion process that restores structure in data, yielding a highly flexible and tractable generative model of the data.

![diffusion_process](/images/tp-3/diffusion_process.png)
*<center><small>Right to Left: Forward diffusion process. Left to Right: Reverse diffusion process</small></center>*

So basically, this approach uses a predefined step that gradually adds noise to an image until we get only pure noise. The method then learns to reverse this noise process to get back to the original image, so it is a denoising process. Ultimately, this method uses a pure noise image as a starting point and reverses the noise until we get an image that makes sense.

To explain further why this is related to "equilibrium statistical physics": The noise we start with can be visualized as a field of particles that are completely disorganized (non-equilibrium) at the beginning but want to find the point where they are organized (equilibrium) and they can only follow the rules they have at their disposal to reach this equilibrium, which are the laws of physics in this case. It's like waiting for a kettle of boiled water to cool down, where the water particles are already heated, so they bounce everywhere, the water just goes naturally to a more stable state of being still and cold by following the laws of thermodynamics.

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/N4LBqY7xPGQ?si=HRiAJziXW8EzXhOE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

*<center><small>This is equivalent to the Forward diffusion process in this analogy</small></center>*

In our case, the "non-equilibrium" is the noise image, the "laws of physics" will be the learned reverse diffusion process and the "equilibrium" state to which it will converge to an image that makes sense.

### Let's Get Into The Details

A (denoising) diffusion model isn't that complex if you compare it to other generative models such as Normalizing Flows, Generative Adversarial Networks (GANs) or Variational AutoEncoders (VAEs): They all convert noise from some a simple distribution, such as Gaussian noise, to a data sample, such as an image. Basically, they're all neural network methods that **learn to gradually denoise data** starting from pure noise.

As we've said, there are 2 processes in diffusion models:
- Forward Diffusion: The predefined process $q$ of adding Gaussian noise to an image until we get only 100% (Gaussian) noise
- Reverse Diffusion: A learned process of denoising an image, where you start from a Gaussian noise and try to reverse (more like rearrange) the noise until you get an image

Both processes are time-based, so they happen in a predefined number of steps $T$, the paper uses $T = 1000$ which is actually a huge number nowadays where we can do under $10$ steps.

#### Forward Diffusion Process

At $t = 0$ you take an image $x_0$ from your train set and then apply to it Gaussian noise in an appropriate way (following a schedule which we'll discuss later) at each time step $t$. By the end, $T = 1000$ or before if you're lucky, you'll end up with a noise image, this noise will actually have converged to what's called an isotropic Gaussian distribution. This specific type of noise has properties that make it much simpler to manipulate in the equations that follow, as opposed to anisotropic distribution. Read more about this [here](https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic) if you're interested.

Let's describe the distribution of our training image dataset as $q(x_0)$ from which we can sample, basically get images from, this way: $x_0 \sim q(x_0)$

Let's also define the Forward diffusion process at each time step $t$: $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1},\beta_t \mathbf{I})$, where $\beta_1,...,\beta_T$ ($0 < \beta_t < 1$) is called the variance schedule because it'll modulate the variance of the Gaussian distribution we are sampling from, making it bigger each time step (meaning there is one $\beta_t$ for each time step), meaning we'll have more spread out noise as the time step $t$ increases.

$q(x_t|x_{t-1})$ just means, from the previous image we added noise to, with $x_0$ being the original image without noise, what is the next image with one noise step added to. $\mathcal{N}$ is a Normal distribution, also called a Gaussian distribution, which has two parameters: the mean $\mu$ and variance $\sigma^2 \geq 0$ (Read more [here](https://en.wikipedia.org/wiki/Normal_distribution#Notation) in case you forgot). A nice trick to get the complex Gaussian noise sample we need above from a simple **standard normal distribution** $\mathcal{N}(\mathbf{0}, \mathbf{I})$ with mean $\mu = 0$ and variance $\sigma = 1$ is to get a sample from this simpler Gaussian distribution and then center and scale it by doing this: $x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} \mathbf{\epsilon}$, where $\mathbf{\epsilon} = \sqrt{1 - \beta_t} x_{t-1}$.

This was for going from one time step to another. The entire process can be summarized as a [Markov chain](https://en.wikipedia.org/wiki/Markov_chain) that gradually adds Gaussian noise to the data: $q(x_{1:T}|x_0) = \prod_{t=1}^{T} q(x_t|x_{t-1})$.
This just means that each step depends on the previous one for the next result, like a chain.

#### $\beta$-Variance Schedule

As we said, the $\beta_t$ are different at each time step, but what values should we assign each?

The original DDPM paper use a linear schedule, basically just a linear function that increases from $B_1 = 10^{-4}$ to $\beta_T = 0.02$.

![beta_linear_schedule](/images/tp-3/beta_linear_schedule.png)
*<center><small>$\beta$ linear schedule</small></center>*

These constants were chosen to be small relative to data scaled to $[−1, 1]$, ensuring that reverse and forward processes have approximately the same functional form while keeping the signal-to-noise ratio at $x_T$ as small as possible.

![beta_linear_schedule_noise_variance](/images/tp-3/beta_linear_schedule_noise_variance.png)
*<center><small>$\beta$ linear schedule noise effect. We see that it simply grows linearly</small></center>*

#### Reverse Diffusion Process

![diffusion_process](/images/tp-3/diffusion_process.png)
*<center><small>Right to Left: Forward diffusion process. Left to Right: Reverse diffusion process</small></center>*

Looking back again at the figure above, we see that we actually go back from noisy to less noisy in the reverse process. If we were able to find $p_{\theta(x_{t-1|t})}$, which knowing the noisier image at $x_{t-1}$ could get us a less noisy image $x_{t}$, would be amazing (you'll notice that indices are reversed compared to $q(x_t|x_{t-1})$, since this is the reverse process). Unfortunately, it's impossible to get a closed form formula that will give us the solution, so to solve that we'll simply use a neural network to approximate this conditional probability distribution $p_{\theta(x_{t-1|t})}$ with $\theta$ being the parameters of the neural network, updated by usual gradient descent.

If we try to deconstruct the process of learning this probability distribution function to simplify things for us, we could assume that the reverse process is also Gaussian, similarly to the forward process. A Gaussian distribution has, again, a mean parameter $\mu$ and a variance $\\Sigma$. Let's write what this function would look like so far: $p_{\theta(x_{t-1|t})} = \mathcal{N}(x_{t-1}; \mu_\theta(x_{t},t), \Sigma_\theta (x_{t},t))$, where we add $\theta$ subscripts to each because we'll want to learn them with our neural network parameters and they'll take as input the current noisy image $x_t$ and the current time step $t$. 
This is amazing because we simplified the problem to just learning the mean and variance parameters of a Gaussian distribution, can't get simpler right?

Actually it does!

The authors of the paper decided that they'll keep the variance fixed and let the neural network learn only the mean $\mu_{\theta}$, so the variance above becomes: $\Sigma_\theta (x_{t},t) = \sigma^2_t \mathbf{I}$ and they set $\sigma^2 = \beta_t$, we come back to our $\beta_t$ constants from above!

<notequote>
Later research shows that it's actually better to learn the variance too during the backward diffusion process, so newer diffusion model architectures also learn both mean and variance.
</notequote>

#### What About the Loss Function?

*This section is the complex part and will defer most of the details to different papers and articles for those who'd like to understand the details more in-depth.*

According to the paper it's all simple:

![loss_ddpm_1](/images/tp-3/loss_ddpm_1.png)

But what's this "usual variational bound on negative log likelihood"? Basically, it's an observation that the forward $q$ and reverse $p_{\theta}$ processes can be seen as a Variational Auto-Encoder (VAE) (first described in the paper "Auto-Encoding Variational Bayes", [Kingma et al., 2013](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fabs%2F1312.6114) for more details). This formulation allows us to use the "Variational lower bound" which basically helps us approximate the log-likelihood of the train data using gradient-based methods by finding a lower bound to this log-likelihood. This tells us that $log(p_{\theta}(x_0)) > L$, where $L$ is the equation above from the paper. Meaning that if we maximize the negative of it $-log(p_{\theta}(x_0))$ through optimization then we'll find a solution to our problem. Why maximizing? Because it'll just bring us in the correct direction to our solution if we move to this bound that they found, even if the solution isn't necessarily the actual maximum.

We're done, right? Not really...

The above is still too complex to compute, we need to decompose further. We already said above that in the forward case we are setting $\beta_t$ as constants, so that's one simplification already. Now, let's simplify further by doing a reparametrization trick by setting $\alpha_t = 1 - \beta_t$ and $\overline{\alpha_t} = \prod_{s=1}^{t} \alpha_s$. This will enable us to rewrite the forward process $q$ as:

$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\overline{\alpha_t}}x_0,(1 - \overline{\alpha_t}) \mathbf{I})$

What do you notice? We were able to rewrite the forward process $q$ so that instead of having to loop from $x_0$ to $x_1$, ..., $x_t$, we can do it by simply using our first image $x_0$ and simply doing a product of the $\alpha_s$ from $s=1$ to $s=t$ and giving it to our forward process. Basically, this shows that we can just sample Gaussian noise and scale it appropriatly (using the $\alpha_s$) and add it to $x_0$ to get $x_t$ directly. Let's not forget that the $\alpha_s$ are just $\alpha_t = 1 - \beta_t$, so a function of $\beta_t$ which we already precalculated above with our linear schedule.

<notequote>
This property will allow us to optimize random terms of the loss function L since we can now get to any time step t from t=0
</notequote>

The paper also decomposes even more the proposed loss to make it more efficient:

![loss_ddpm_2](/images/tp-3/loss_ddpm_2.png)

We see they decompose it into $L = L_0 + L_1 + ... + L_T$ and that each $L_t$ term except for $L_0$ are a KL-divergence (or  [Kullback–Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) is a function that computes the difference or distance between two probability distributions) between $q$ and $p_\theta$ which are two Gaussian distributions. The loss can then be rewritten as an L2-loss by using the means of each distribution.

Finally, they do one final reparametrization on the mean that enables the network to predict the added noise on an image instead of of predicting the mean itself. This means that the network actually becomes a noise predictor rather than a Gaussian mean predictor. The mean reparametrization looks like this:

![reparam_mean](/images/tp-3/reparam_mean.png)

Plugging this into the Gaussian probability density function (PDF) leads us to the following loss function:

$L = || \mathbf{\epsilon} - \mathbf{\epsilon}_\theta(x_t, t) ||^2 $

$L = || \mathbf{\epsilon} - \mathbf{\epsilon}_\theta( \sqrt{\bar{\alpha}_t} x_0 + \sqrt{(1- \bar{\alpha}_t)  } \mathbf{\epsilon}, t) ||^2.$

<notequote>
There should be constants left, on the left side of the formula, that depend on β but the authors just removed them to simplify the final loss and constants don't usually affect things much.
</notequote>

*If you want more more details on the calculations, see [section 3.2 of the paper](https://arxiv.org/pdf/2006.11239).*

$x_0$ being the initial unmodified image, $\mathbf{\epsilon}$ is the noise level at time step $t$ and $\mathbf{\epsilon}_\theta (x_t, t)$ is our neural network.

After all this simplification we just end up with a simple Mean Squared Error (MSE) between the true noise $\mathbf{\epsilon}$ and the Gaussian noise predicted by our neural network $\mathbf{\epsilon}_\theta (x_t, t)$.

The final training procedure is:

![ddpm_training](/images/tp-3/ddpm_training.png)

In other words:
* We take a random image $x_0$ from the train dataset $q(x_0)$
* We sample a noise level $t$ uniformally between $1$ and $T$, meaning we want it to be at a random time step
* We sample some noise from a Gaussian distribution and corrupt the input by this noise at level $t$
* The neural network is trained to predict this noise based on the corruped image $x_t$. It'll try to find the noise that was applied on $x_0$ based on the fixed schedule $\beta_t$.

That was a lot of reparametrization but we got there, now let's try to understand the procedure to generate an image.

### Generating an Image by Sampling

Generating an image is actually quite simple (relatively to what we just saw). We say that we "sample" an image because we get it from a distribution. In our case, we did all this hard work to be able to work with Gaussian distributions, so let's use it.

The process to sample an image is basically to start from a pure noise image, reverse the noise as much as we can until we end up with a clean image.

![reverse_diffusion](/images/tp-3/reverse_diffusion.png)
*<center><small>Forward/Reverse diffusion process</small></center>*

To sample from a standard Gaussian (or Normal) distribution we just do $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and since we want to sample from our own Gaussian distribution for which we learned the mean and variance parameters, let's scale our samples: $\mathbf{z_{\theta, t}} = \mathbf{\mu_\theta(x_t, t)} + \sigma_t \mathbf{z}$, and $\mathbf{z_{\theta, t}}$ is our approximate to $x_{t-1}$ having started from $t=T$, we want to go to $t=0$ using repetitive sampling until we get $x_0$, a real image.

In our case: $x_{t-1} = \frac{1}{\sqrt(\alpha_t)}(x_t - \frac{\beta_t}{\sqrt{1 - \overline{\alpha_t}}} \epsilon(x_t, t) ) + \sigma_t z$, where $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ from our simple Gaussian distribution. 

We end up with the following algorithm for generating an image:

![ddpm_sampling](/images/tp-3/ddpm_sampling.png)

We finally have the full algorithm to start implementing our new model!


### Something to Note

Diffusion Models also exist under different forms mathematically speaking that all have different names (like Score-based generative models, SDE-based generative models, Langevin dynamics, etc) that people discovered at different times. To learn more, read [this](http://yang-song.net/blog/2021/score/).

Also, some research ("Cold Diffusion: Inverting Arbitrary Image
Transforms Without Noise", [Bansal et al. 2022](https://arxiv.org/abs/2208.09392)) has been done to actually show that diffusion models don't need to invert actual noise but can actually invert any arbitrary function like blur, masking, snowification (adding a snow like effect) or even animorphosis (transforming into an animal)!

![cold_diffusion](/images/tp-3/cold_diffusion.png)

## Let's Implement This!

Ok, we went quite deep in the theory, but we're finally at the implementation phase. Let's train a model to generate some images!

First, clone the repository found [here](https://github.com/cpcdoy/dl_practical_work) and you'll find many helper files in the `practical_work_3_ddpm/` folder and you will be using the helper functions in there and importing them in your notebook, or in your Python files, up to you.

### Loading and Process the Dataset

Let's generate images from the [Fashion MNIST dataset](https://huggingface.co/datasets/fashion_mnist). It's basically the equivalent of the handwritten digit dataset MNIST but for clothes, so it's a more complex dataset. There are 10 classes of clothes and the images are grayscale and $28x28$ in size to match the original MNIST dataset.

![fashion_mnist](/images/tp-3/fashion_mnist.png)
*<center><small>Examples from Fashion MNIST</small></center>*

Let's use 🤗 Datasets to load [this dataset](https://huggingface.co/datasets/fashion_mnist) very easily:

```Python
from datasets import load_dataset

dataset = load_dataset("fashion_mnist") # load dataset from the hub
image_size = 28 # Height = Width = 28
channels = 1 # Grayscale image so 1 channel1
batch_size = 128 # Works well on Google Colab with 16GB VRAM but modify this to match your hardware
```

The paper specifically says that each pixel values are converted from the $[0, 255]$ range into $[-1,1]$. They also apply random horizontal flips to the image to augment the data and say it increases "quality slightly", so we'll do it too:

```Python3
from torchvision import transforms
from torch.utils.data import DataLoader

# Define the image transformations using torchvision
transform = Compose([
            transforms.RandomHorizontalFlip(), # Horizontal flips augmentation
            transforms.ToTensor(), # Convert to PyTorch tensor
            transforms.Lambda(lambda t: (t * 2) - 1) # Convert to the [-1, 1] range
])

# Transform function
def transforms(examples):
   # Apply the transformation on each train image
   # "image" is the key that contains the image data in the Fashion MNIST dataset
   examples["pixel_values"] = [transform(image) for image in examples["image"]]

   # We remove it since we transformed the image above to what fits us and we don't need the original image anymore
   del examples["image"]

   return examples

# Apply the transforms
transformed_dataset = dataset.with_transform(transforms).remove_columns("label")

# Create the dataloader that will output batches of our transformed train data
dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)
```

### The Model Architecture

*Wait, we haven't talked about the model architecture at all, have we?*

To understand the neural network required for this task, let's break it down step-by-step. The network needs to process a noisy image at a given time step and return the predicted noise in the image. This predicted noise is a tensor with the same dimensions as the input image, meaning the network's input and output tensors have identical shapes. So, what kind of neural network is suited for this?

Usually, a network architecture called an *Autoencoder* is used here. Autoencoders feature a "bottleneck" layer between the encoder and decoder. The encoder compresses the image into a smaller hidden representation, and the decoder reconstructs the image from this representation. This design ensures that the network captures only the most crucial information in the bottleneck layer.

<notequote>
The "bottleneck" is called like that because it's a place in the network which can't encode a lot of information and this forces the network to compress efficiently the information if it wants to reuse it or even reconstruct it later!
</notequote>

In terms of architecture, the authors of the DDPM paper opted for an architecture that follows the backbone of [PixelCNN++](https://arxiv.org/abs/1701.05517), which is a U-Net introduced by [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597) (but based on a [Wide ResNet](https://arxiv.org/abs/1605.07146)). This architecture achieved state-of-the-art results in medical image segmentation at the time. Being an Autoencoder, a U-Net has a bottleneck that helps the network learn essential features. Additionally, it includes residual connections between the encoder and decoder, inspired by [He et al., 2015](https://arxiv.org/abs/1512.03385)'s ResNet, to improve gradient flow.

<questionnote>
What are possible use cases of an Autoencoder architecture?
</questionnote>

![u_net](/images/tp-3/u_net.png)

As shown in the figure, a U-Net model first downsamples the input image (reducing its spatial resolution) and then upsamples it back to the original size. 

### How Do We Teach Time to Our Model?

As we've seen, the diffusion process is time dependent where we have the value $t$ to help us keep track of time. Each time step $t$ works at a specific noise level, so to help our model operate at these specific noise levels, we'll need to feed it the $t$ value.

To implement this, the authors actually use 

###   

### Let's Implement a U-Net Block



<exercisequote>
Implement the U-Net block described here.
</exercisequote>



### $\beta$-Variance Schedule

The $\beta_t$ linear schedule is actually quite strong because noise grows very rapidly and we even start having issues recognizing the original image with the naked eye.

<exercisequote>
Improve the current scheduling by implementing cosine scheduling as proposed above.
</exercisequote>


# Bonus Exercise

<exercisequote>
Train this model on your own dataset!
</exercisequote>

Here's a list of simple datasets:

- CIFAR10

![cifar10_sample](/images/tp-3/cifar10_sample.jpg)
*<center><small>Example image from CIFAR10: A cat yawning</small></center>*

- CelebA
- LSUN

These datasets are more complex and will require you have a better GPU or that you're able to run it on Google Colab for 1 day straight on their Free Tier T4 GPUs.

---

# You're Done!

Great job, you made it this far!

## Class Students

Send it on the associated MS Teams Assignment.

## Anyone else

Send it to my [email adress](mailto:chady1.dimachkie@epita.fr?subject=TP%203) with the subject **Practical Work 3**: [chady1.dimachkie@epita.fr](mailto:chady1.dimachkie@epita.fr?subject=Practical%20Work%203)

**Don't hesitate if you have any questions!**

→ [Coming Next: Practical Work 4](/articles/)