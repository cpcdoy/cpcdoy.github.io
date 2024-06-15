---
title: 4. How to Read a Machine Learning Research Paper
date: 2024-06-11
images:
- /images/tp-4/paper_reading.jpg
---

In this third practical work, we'll be learning **How to Read a Machine Learning Paper**.

# Let's Get Started


## Abstract

As someone studying machine learning, you'll have to read a technical paper at several points in your life to review the current literature on the specific topic you want to research. You've actually already come across several papers while doing the previous [practical works](/articles/).

Doing a *literature review* is essential when starting to work on a subject: People have probably already looked into the topic you want to study, so learning from their previous work will help you save time and reach a solution faster.

However, reading papers in ML is not always simple, since ML is a complex field applied to many different domains, you'll have to have previous knowledge of mathematics to understand the formulas, you'll need to have a background in programming to be able to implement the paper (ML is literally **Machine** Learning) and on top of all that, you might need to learn some of the specifics of the field your paper is being applied to: e.g. It could be an ML paper applied to the medical field (e.g. The U-Net foundational architecture was introduced in a [Biomedical paper](https://arxiv.org/abs/1505.04597)) or to finance.


Also, reading a paper can take you time at the beginning since you'll be lacking many of the concepts and tools to reading them efficiently, so don't give up, you'll get there!

In this practical work, you'll learn how to read a paper fast and get the important information that'll help you.


## Introduction

As we've said, reading a paper isn't always simple business, but it becomes easier the more papers you read. On top of that, at the beginning you'll be lost at what to look for in a paper, how to read it and what you should expect from it.
To help you at the beginning, think that when reading a paper you'll need to have a goal in mind and you should be able to answer some questions about the topic you're trying to learn about.

Let's discuss a few of these questions!

### Where's the research currently at on the topic?

Some topics can be cutting edge and can still be in early research, for example, [explainability of language models](https://arxiv.org/abs/2309.01029) is a field that has a few years and while a lot of research is currently being done, no real "fits all" solution actually exists because it's a very hard problem to solve: Why did a model output this specific answer?

### What are the challenges typically encountered?

Each sub-field of machine learning has its own challenges: For example, for a while in language modeling and generation (GPT-like models), the challenge was to get a model that would output something coherent for more than one sentence. Early research was only able to output one sentence at best and the follow-up sentences would make no sense or be random even if they syntaxically made sense. Nowadays, there are other problematics of making language models generate answers much faster and using less memory.

In summary, after reading a few papers on your topic, you need to be able to get a better understanding of current challenges that need to be solved.

### What are the best solutions available?

In a paper, you'll typically find a few paragraph towards the beginning of the paper (In sections like `Background` or even in the `Introduction`) that will summarize the current state of the field and what the research themselves have looked into before doing this research. In this part, they'll explain what are/were the current best methods and what their methods is improving on top of the others. This is also a good opportunity to read about these competing methods they mention.

On top of that, you'll typically find benchmarks in the middle or towards the end of the paper against other methods they've tried. Typically the paper will show improved results, but be wary that sometimes you can make numbers say what you want ;)

### Is there any usable code in the paper?

Something you should look for is: Is there any available code or even a usable product (e.g. a tool, framework, library, etc) out of this research paper?

This will definitely simplify your life if you're looking to use a method quickly instead of having to reimplement it. Make sure to look at the license of the code, depending on what you want to do with it (e.g. commercial usage, etc) you might run into some blockers.

### What are the limitations of the current solutions and what could be improved in the future?

When you're reading a paper, the paper never fully solves a problem and then marks the end of the current field of research. There are always limitations or things to improve that you'll need to spot because it might not always be explicitely stated.

Often, these limitations can be discussed towards the end of the paper or in the conclusion itself. So be sure to look for this. Also, sometimes the researchers won't mention these limitations or haven't yet found them. 

<notequote>
You should know that some research labs are also driven by economical factors. So sometimes they'll make their research look nicer than it actually is. Scores without any way to reproduce them can mean anything!
</notequote>

## Types of papers

Now that you finally know what to ask yourself when reading a paper, you also need to choose the actual paper you want to read. For that you'll need to know that there a several types of papers that can have different layouts, lengths and types of things they discuss. Here are a few of them:

### Papers Introducing a New Method

These papers will usually be either a breakthrough new architecture or a novel way of doing something that already exists.

Examples:
  - [YOLO](https://arxiv.org/abs/1506.02640) for fast object detection
  - [Transformer](https://arxiv.org/abs/1706.03762) architecture for more accurate language modeling
  - [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929), as its name implies, applies the Transformer architecture but to Computer Vision tasks. An alternative to Convolutional Neural Networks (CNNs).
  - [Retentive Network](https://arxiv.org/abs/2307.08621), which is a different way of doing language modeling and tries to be a successor of the Transformer (yet remains to be seen in practice)

### Papers Showing Incremental Improvements

Research always works in incremental improvements, even if we see breakthrough methods here and there, they often build upon a long list of incremental improvements.

Examples of improvements on [YOLO](https://arxiv.org/abs/1506.02640):
- [YOLO9000](https://arxiv.org/abs/1612.08242), [YOLOv3](https://arxiv.org/abs/1804.02767), [YOLOv4](https://arxiv.org/abs/2004.10934), [yes they skipped v5](https://github.com/ultralytics/yolov5/issues/11381), [YOLOv6](https://arxiv.org/abs/2209.02976), [YOLOv7](https://arxiv.org/abs/2207.02696), [YOLOv8 paper is not yet ready](https://github.com/ultralytics/ultralytics/issues/204), [YOLOX](https://arxiv.org/abs/2107.08430), [PP-YOLOE](https://arxiv.org/abs/2203.16250) and more!

### Papers Introducing a New Framework

Frameworks like PyTorch or Tensorflow can seem like a common tool, but they are actually the result of state-of-the-art research on commoditizing machine learning into powerful, efficient and simple tools to use. So they have their own papers:
  - [PyTorch](https://arxiv.org/pdf/1912.01703), the machine learning library we've been using so far.
  - [TensorFlow](https://arxiv.org/abs/1605.08695), an alternative to PyTorch.

### Making a survey of a specific field

Some fields have so much research happening that even if you're working directly in the field you can struggle of keeping track of everything. Or if you're from a different field and want to understand the current state of the field, survey papers are quite convenient for that.

A few examples of surveys:
  - [Large Language Models: A Survey](https://arxiv.org/abs/2402.06196): Survey on the current state of Large Language Models (LLMs) from dataset creation to training to evaluation and metrics and latest trends in the field
  - [Deep Learning for Image Super-resolution: A Survey](https://arxiv.org/pdf/1902.06068): Unfortunately, it can sometimes be hard to find an up to date survey paper, like this one dating from 2019 and was last updated in 2020. Methods have changed quite a lot since, but it's still very useful to get a history on how things were just a few years ago. You can then start looking for relevant papers with a solid starting point.

### Papers Introducing a Metric

Some papers introduce new metrics to better evaluate model results. This is very important because the metrics we have never show the full picture of your model's behavior.

A few examples of papers introducing datasets:
  - [Fréchet Inception Distance](https://arxiv.org/abs/1706.08500) (FID) is a metric that improves how similar generated images are to real ones.
  - [Focal Loss](https://arxiv.org/abs/1708.02002) is a loss function that tries to help in the case of imbalanced data. It was introduced in the context of improving object detection methods but can actually be used in many other contexts since it is a modification of the Cross-Entropy loss.
- Publishing a new dataset:
  - [Microsoft COCO](https://arxiv.org/abs/1405.0312)
  - [CIFAR-100](https://paperswithcode.com/dataset/cifar-100)
  - [MultiNLI](https://arxiv.org/abs/1704.05426)
  - [The Pile](https://arxiv.org/abs/2101.00027)

### Papers That Break or Confirm a Common Belief

Other types of fun papers are the ones that try to contradict or confirm a common belief. Basically, anything that people take for granted and that's always been there and that no most haven't thought of questioning.

A fun example is the paper [Attention is not Explanation](https://arxiv.org/abs/1902.10186) followed very shortly by [Attention is not not Explanation](https://arxiv.org/abs/1908.04626), who's right? 😃

### Bonus

There are also just papers that play on the hype of a specific paper name like the famous foundational paper [Attention is All You Need](https://arxiv.org/abs/1706.03762) and use it to make their title more appealing like [Attention Is Not All You Need Anymore](https://arxiv.org/abs/2308.07661). Actually there's a lot more, take a look at [this list](https://github.com/KentoNishi/awesome-all-you-need-papers). You never know, maybe the content is as original as the title.

Finally, there are papers that are a bit fun, unusual or try to pass a message, for example the YOLOv3 paper: [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767). I'll let you read the intro and conclusion to make up your own mind...

<div>

  <center><figure class="half" style="display:flex">
      <img style="width:300px; margin-right: 10px; margin-left: -150px" src="/images/tp-4/yolov3_intro.png">
      <img style="width:700px" src="/images/tp-4/yolov3_conclusion.png">
  </figure></center>

  <div class="sticky" id="stickyElement">
    <img src="/images/tp-4/pixel_shuffle_paper_border.png" alt="paper">
  </div>

</div>

## How do we Actually Read a Paper Fast?

<h3>← Look at this example paper</h3>

<notequote>

You've already seen this paper in the previous practical works. It's the **[Pixel Shuffle](https://arxiv.org/abs/1609.05158)** paper!

</notequote>

We'll use the [Pixel Shuffle](https://arxiv.org/abs/1609.05158) to demonstrate how to read a paper effectively. Which you've already look at (in theory :D).


### The First Pass: Title → Abstract → Figures

<div>

  <div class="sticky" id="stickyElement">
    <img src="/images/tp-4/pixel_shuffle_paper_border_page_reading_order.png" alt="paper">
  </div>

In the first pass of reading your paper, you'll read it very roughly, you really want to see if it looks like it'll be useful to you and if the results look interesting. In papers you'll often find the abstract contains a lot of information and summarizes the paper quite well. The figures such as graphs, images or benchmark results will show you the architecture if there is a model, the results of their benchmarks, comparison with other methods and more.

Let's look at the first page of our paper to start the first pass:

<span style="color: blue"> 1. </span> The title: Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
  - Real-Time: Means this model will be fast and usable in contexts where you need instant output ($\le 100$ milliseconds) from your model
  - Single Image: It means it works only one image at a time. Some models use several images, it could be pictures taken less than a second apart or frames from a video. This usually helps get better quality and this what your phone does when you take a picture for example, it takes a burst and then from the small jitters caused by your hands will actually get a lot of subpixel information from all frames and use it to reconstruct a higher quality, less noisy image even in low light conditions.
  - Video: This means that this paper applies super resolution on video data. We now understand why real-time is needed, since you don't want to wait for hours to process your video.
  - Super-Resolution: The process of taking a lower resolution image and reconstructing details to get a higher resolution image. This is what we did in [practical work 2](/articles/cv/tp-2/).
  - Using an Efficient Sub-Pixel: Here we understand that this paper tries to work at the subpixel level, meaning directly either on the components of a pixel (RGB or any other color space) or on the latent representation of these components. We'll only find out later by reading the paper, but this is only a detail for now.
  - Convolutional Neural Network: The model will be based on a [Convolutional Neural Network (CNN)](/articles/cv/tp-2/), not too surprising.

<span style="color: #FAD5A5"> 2. </span> The people who contributed: </div>  
  - You can give it a very quick glance, and where they work. Here they all work at the social media Twitter (now called *X*). So we know they have the budget of a tech company to make interesting research that'll possibly be profitable to them.
  - Another fact about this is that the paper is usually ordered by the authors who wrote the paper. The first author is usually the one who contributed most. Some papers also sometimes add more detail on the contributions of each person.
  - If you find the paper interesting, the authors might have written other similar quality work, so it can be good to look up their name later.


<notequote>
You can also ask yourself, why Twitter would need this kind of research? Imagine the quantity of data they store for the images shared on social media. Reducing this by a factor of 1.5x or 2x would be a huge cost saver for them.
</notequote>

<div>

  <div class="sticky" id="stickyElement">
    <img src="/images/tp-4/pixel_shuffle_paper_abstract_p1.png" alt="paper">
  </div>

  <span style="color: red"> 3. </span> The Abstract: This is the part where you need to focus and extract as much information as you can. Let's decompose it: </div> <div>
  - You can also ask yourself, why Twitter would need this kind of research? Imagine the quantity of data they store for the images shared on social media. Reducing this by a factor of 1.5x or 2x would be a huge cost saver for them.You can also ask yourself, why Twitter would need this kind of research? Imagine the quantity of data they store for the images shared on social media. Reducing this by a factor of 1.5x or 2x would be a huge cost saver for them.
  
  <div style="margin-bottom: 50px" class="sticky" id="stickyElement"> <img src="/images/tp-4/pixel_shuffle_paper_abstract_p2.png" alt="paper"> </div>
  
  - You can also ask yourself, why Twitter would need this kind of research? Imagine the quantity of data they store for the images shared on social media. Reducing this by a factor of 1.5x or 2x would be a huge cost saver for them.You can also ask yourself, why Twitter would need this kind of research? Imagine the quantity of data they store for the images shared on social media. Reducing this by a factor of 1.5x or 2x would be a huge cost saver for them.You can also ask yourself, why Twitter would need this kind of research? Imagine the quantity of data they store for the images shared on social media. Reducing this by a factor of 1.5x or 2x would be a huge cost saver for them.You can also ask yourself, why Twitter would need this kind of research? Imagine the quantity of data they store for the images shared on social media. Reducing this by a factor of 1.5x or 2x would be a huge cost saver for them.You can also ask yourself, why Twitter would need this kind of research? Imagine the quantity of data they store for the images shared on social media. Reducing this by a factor of 1.5x or 2x would be a huge cost saver for them.You can also ask yourself, why Twitter would need this kind of research? Imagine the quantity of data they store for the images shared on social media. Reducing this by a factor of 1.5x or 2x would be a huge cost saver for them.You can also ask yourself, why Twitter would need this kind of research? Imagine the quantity of data they store for the images shared on social media. Reducing this by a factor of 1.5x or 2x would be a huge cost saver for them.You can also ask yourself, why Twitter would need this kind of research? Imagine the quantity of data they store for the images shared on social media. Reducing this by a factor of 1.5x or 2x would be a huge cost saver for them.

  </div> <div> <div style="margin-bottom: 70px" class="sticky" id="stickyElement"> <img src="/images/tp-4/pixel_shuffle_paper_abstract_p3.png" alt="paper"> </div>

  - You can also ask yourself, why Twitter would need this kind of research? Imagine the quantity of data they store for the images shared on social media. Reducing this by a factor of 1.5x or 2x would be a huge cost saver for them.You can also ask yourself, why Twitter would need this kind of research? Imagine the quantity of data they store for the images shared on social media. Reducing this by a factor of 1.5x or 2x would be a huge cost saver for them.You can also ask yourself, why Twitter would need this kind of research? Imagine the quantity of data they store for the images shared on social media. Reducing this by a factor of 1.5x or 2x would be a huge cost saver for them.You can also ask yourself, why Twitter would need this kind of research? Imagine the quantity of data they store for the images shared on social media. Reducing this by a factor of 1.5x or 2x would be a huge cost saver for them.
  
  </div> <div> <div class="sticky" id="stickyElement"> <img src="/images/tp-4/pixel_shuffle_paper_abstract_p4.png" alt="paper"> </div>

  - You can also ask yourself, why Twitter would need this kind of research? Imagine the quantity of data they store for the images shared on social media. Reducing this by a factor of 1.5x or 2x would be a huge cost saver for them.You can also ask yourself, why Twitter would need this kind of research? Imagine the quantity of data they store for the images shared on social media. Reducing this by a factor of 1.5x or 2x would be a huge cost saver for them.You can also ask yourself, why Twitter would need this kind of research? Imagine the quantity of data they store for the images shared on social media. Reducing this by a factor of 1.5x or 2x would be a huge cost saver for them.You can also ask yourself, why Twitter would need this kind of research? Imagine the quantity of data they store for the images shared on social media. Reducing this by a factor of 1.5x or 2x would be a huge cost saver for them.

  </div>



## It's Your Turn!


<exercisequote>

Read one of the following papers and take notes. On top of that, you'll have to explain the order in which you read the paper and your thought process, things you didn't understand, the research you didn't externally to the paper if you were confused, etc. **You'll be sending me a written report as your practical work!**

</exercisequote>