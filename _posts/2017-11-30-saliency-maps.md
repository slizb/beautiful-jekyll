---
layout: post
title: Where the Machines are Looking
subtitle: Visualization of neural networks using saliency maps.
image: /img/kid_backpack_thumb.png
tags: [computer-vision, deep-learning]
---

# *** THIS POST IS UNDER CONSTRUCTION ***
# *** PLEASE DO NOT SHARE ***

#### *What is this about?*
In this post, I walk through the use of saliency maps to visualize the decision-making process of a neural network model. 
#### *Why should I care?*
Neural networks are often considered to be black-box models, whose decisions are made behind a mysterioius veil. While their 
decision making process can indeed be complex, its not all hidden to us, and saliency maps are here to help make sense of it. 
#### *Want to learn more?*
This blog post is awesome, but you'll learn a lot more from a book than from me... If you're interested in learning more about 
computer vision, check out these books:

## Background
One of the biggest critiques of deep learning models is that they have limited interpretability. It turns out that [multiple techniques have been developed](http://cs231n.github.io/understanding-cnn/) specifically to visualize how they make decisions. One such technique is known as saliency mapping, discussed by Karen Simonyan et al. in their paper [*Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps*](https://arxiv.org/pdf/1312.6034.pdf). In their paper, they describe saliency like this:
> Another interpretation of computing the image-specific class saliency using the class score derivative
> is that the magnitude of the derivative indicates which pixels need to be changed the least to affect 
> the class score the most. One can expect that such pixels correspond to the object location in the image. 

In other words, saliency can be computed for a given image, and a given class. It tells us which pixels in the image contribute most to the model's classification of that class. Cool!

There happens to be a great python package by Raghavendra Kotikalapudi called [*keras-vis*](https://github.com/raghakot/keras-vis) that supports saliency map visualization for Keras models. Lets try it out!

## Demo
First, we need a model to test out. For simplicity, let's just use a pre-trained model... I choose ResNet:

```python
from keras.applications.resnet50 import ResNet50

model = ResNet50()
```

Next, according to Kotikalapudi, we need to switch the softmax activation out for linear or the results might be suboptimal, since the gradient of the output node will depend on all the other node activations. Doing this in keras is tricky, so he provides `utils.apply_modifications` to make it easy. 

```python
from vis.utils import utils
from keras import activations

model.layers[-1].activation = activations.linear
model = utils.apply_modifications(model)
```
Let's pick an image to classify. ResNet was trained on the ILSVRC data (Imagenet), so we need to be sure that the image we choose depicts one of the classes from ILSVRC. I know that some of the classes are for cats, so lets try out this cute kitten:

