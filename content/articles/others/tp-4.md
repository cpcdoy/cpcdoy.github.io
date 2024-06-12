---
title: 4. How to Read a Machine Learning Research Paper
date: 2024-06-11
images:
- /images/tp-3/diffusion_model_arch.png
---

In this third practical work, we'll be learning **How to Read a Machine Learning Paper**.

# How to Read a Machine Learning Paper?

## Abstract

As someone studying machine learning, you'll have to read a technical paper at several points in your life to review the current literature on the specific topic you want to research. You've actually already come across several papers while doing the previous [practical works](/articles/).

Doing a *literature review* is essential when starting to work on a subject: People have probably already looked into the topic you want to study, so learning from their previous work will help you save time and reach a solution faster.

However, reading papers in ML is not always simple, since ML is a complex field applied to many different domains, you'll have to have previous knowledge of mathematics to understand the formulas, you'll need to have a background in programming to be able to implement the paper (ML is literally **Machine** Learning) and on top of all that, you might need to learn some of the specifics of the field your paper is being applied to: e.g. It could be an ML paper applied to the medical field (e.g. The U-Net foundational architecture was introduced in a [Biomedical paper](https://arxiv.org/abs/1505.04597)) or to finance.

Also, reading a paper can take you time at the beginning since you'll be lacking many of the concepts and tools to reading them efficiently, so don't give up, you'll get there!

In this practical work, you'll learn how to read a paper fast and get the important information that'll help you.

## Introduction

When reading a paper, you need to have a goal in mind and you should be able to answer some questions about the topic you're trying to learn about:

- Where's the research currently at on the topic?
- What are the challenges typically encountered?
- What are the best solutions available?
  - This also means: Is there any available code or even a usable product (e.g. a tool, framework, library, etc) out of this research paper?
- What are the limitations of the current solutions?
- What are the possible solutions that should be explored in the future?

# Types of papers

- New Method: e.g. [YOLO](https://arxiv.org/abs/1506.02640), [Transformer](https://arxiv.org/abs/1706.03762)
- Incremental Improvements: e.g. [YOLO9000](https://arxiv.org/abs/1612.08242), [YOLOv3](https://arxiv.org/abs/1804.02767), [YOLOv4](https://arxiv.org/abs/2004.10934), [yes they skipped v5](https://github.com/ultralytics/yolov5/issues/11381), [YOLOv6](https://arxiv.org/abs/2209.02976), [YOLOv7](https://arxiv.org/abs/2207.02696), [YOLOv8 paper is not yet ready](https://github.com/ultralytics/ultralytics/issues/204), [YOLOX](https://arxiv.org/abs/2107.08430), [PP-YOLOE](https://arxiv.org/abs/2203.16250) and more!
- New frameworks: e.g. [PyTorch](https://arxiv.org/pdf/1912.01703), [TensorFlow](https://arxiv.org/abs/1605.08695)
- Survey: e.g. [Large Language Models: A Survey](https://arxiv.org/abs/2402.06196)
- Introduce metrics: e.g. [Focal Loss](https://arxiv.org/abs/1708.02002)
- New dataset: e.g. [Microsoft COCO](https://arxiv.org/abs/1405.0312), [CIFAR-100](https://paperswithcode.com/dataset/cifar-100), [MultiNLI](https://arxiv.org/abs/1704.05426)
- Papers that break a previous belief: e.g. [Attention is not Explanation](https://arxiv.org/abs/1902.10186), followed very shortly by [Attention is not not Explanation](https://arxiv.org/abs/1908.04626) 😃
- etc

This assumes 

Now, not every paper will be useful or worth your time. It's important to be able to effectively read a paper and see if it'll help you or not. Understanding that the field of ML can sometimes be driver by labs that need to publish and paper counts, etc etc