---
layout: post
title: Make Your Neural Networks Pay Attention!
subtitle: Visual examples and discussion on the uses of object detection and object segmentation with Mask R-CNN.
image: /img/federer.png
tags: [computer-vision, deep-learning]
---

# ***THIS PAGE IS UNDER CONSTRUCTION. PLEASE DO NOT SHARE***

Earlier this year, Kaiming He et al released [their paper *Mask R-CNN* on arXiv](https://arxiv.org/abs/1703.06870). In it, they make 
some impressive claims. Here is their abstract:
> *We present a conceptually simple, flexible, and general framework for object instance segmentation. 
> Our approach efficiently detects objects in an image while simultaneously generating a high-quality 
> segmentation mask for each instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a 
> branch for predicting an object mask in parallel with the existing branch for bounding box recognition. 
> Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, 
> Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same 
> framework. We show top results in all three tracks of the COCO suite of challenges, including instance 
> segmentation, bounding-box object detection, and person keypoint detection. Without tricks, Mask R-CNN 
> outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners. 
> We hope our simple and effective approach will serve as a solid baseline and help ease future research in 
> instance-level recognition. Code will be made available.*

Just this past week, the codebase for *Mask R-CNN* was made public. Since I was in need of a pre-trained object detection model,
I decided to take it for a test drive, and check out its segmentation performance too.
 
