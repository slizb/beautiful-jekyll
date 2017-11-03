---
layout: post
title: Make Your Neural Networks Pay Attention!
subtitle: Visual examples and discussion on the uses of object detection and object segmentation with Mask R-CNN.
image: /img/federer.png
tags: [computer-vision, deep-learning]
---

# ***THIS PAGE IS UNDER CONSTRUCTION. PLEASE DO NOT SHARE***

Earlier this year, Kaiming He et al released [their paper *Mask R-CNN* on arXiv](https://arxiv.org/abs/1703.06870). In it, they make some impressive claims. Here is their abstract:
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

Just this past week, the codebase for *Mask R-CNN* was [made public on GitHub](https://github.com/matterport/Mask_RCNN), and it generated [a bit of buzz on Reddit](https://www.reddit.com/r/MachineLearning/comments/7a7hm1/p_tensorflow_implementation_of_mask_rcnn_for/). Since I was in need of a pre-trained object detection model, I decided to take it for a test drive, and check out its segmentation performance too.

## Installation
Installation was not exactly simple, but also not not *too* complicated. I started with the instructions on the GitHub page, but found I needed a bit more. I went through the steps generally as follows:
1. clone the Mask_RCNN repository
2. download the pre-trained COCO weights (mask_rcnn_coco.h5) from the repository's releases page
3. move the weights to the just created Mask_RCNN directory
4. upgrade tensorflow to >= 1.3
5. clone the special forked version of COCO from https://github.com/waleedka/coco
6. run "make" under coco/PythonAPI
7. run python setup.py under coco/PythonAPI
8. install missing dependencies using conda / pip until setup works
9. try to run demo.ipynb from jupyter and cross fingers
 
