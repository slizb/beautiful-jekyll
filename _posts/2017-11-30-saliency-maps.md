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
Now, let's pick an image to classify. ResNet was trained on the ILSVRC data (Imagenet), so we need to be sure that the image we choose depicts one of its' classes. I know that some of the classes are fcats, so lets try out this cute kitten:

<p align="center">
    <img src="https://slizb.github.io/img/posts/saliency_maps/kitten.jpg" width="500">
</p>

Indeed, when we predict, the top results are cats:

```python

from keras.applications import imagenet_utils 

def make_prediction(img_array):
    x = img_array.copy()
    x = imagenet_utils.preprocess_input(x)
    img_arr_batch = x.reshape((1,) + x.shape)
    soft_preds = model.predict(img_arr_batch)
    prediction = soft_preds.argmax()
    detailed_prediction = imagenet_utils.decode_predictions(soft_preds)
    return prediction, detailed_prediction

pred, top_5 = make_prediction(kitten)

print(top_5)

```
 [[('n02123045', 'tabby', 8.4959364),<br>
   ('n02129165', 'lion', 8.1750641),<br>
   ('n02123394', 'Persian_cat', 7.8045616),<br>
   ('n02123159', 'tiger_cat', 7.1740088),<br>
   ('n02124075', 'Egyptian_cat', 6.7863522)]]

Now let's make a saliency map:

```python

from vis.visualization import visualize_saliency

grads = visualize_saliency(model, layer_idx, filter_indices=true_label, seed_input=kitten)

plt.imshow(kitten_img)
plt.imshow(grads, alpha=.6)
plt.axis('off')
plt.imshow(grads)

```

<p align="center">
    <img src="https://slizb.github.io/img/posts/saliency_maps/kitten_saliency.jpg" width="500">
</p>

Very nice! The saliency map shows us which pixels contribute most to classifying this image as the *'tabby'* class. It looks good; most of the attention of the network is right around the body of the cat.  However, its a bit choppy to my eyes.  Let's smoothe out the result with a gaussian filter:

```python

import scipy.ndimage as ndimage

smoothe = ndimage.gaussian_filter(grads[:,:,2], sigma=5) 
plt.imshow(kitten_img)
plt.imshow(smoothe, alpha=.7)
plt.axis('off')
plt.show()

```

<p align="center">
    <img src="https://slizb.github.io/img/posts/saliency_maps/kitten_gaussian.jpg" width="500">
</p>

Beautiful! The gaussian filter makes the saliency map much more interpretable. Now it is clear that the model is indeed focusing its attention on the cat, which is what we want to see. This example is rather simple, though. How about an image with two classes in it? Or lots of backbround noise? Or context clues? First, let's generalize our workflow with a few helper functions:

```python

def compute_saliency_map(model, array, target_class, layer_idx=-1):
    grads = visualize_saliency(model, layer_idx, filter_indices=target_class, seed_input=array)
    smoothe = ndimage.gaussian_filter(grads[:,:,2], sigma=5) 
    return smoothe

def render_img_on_grid(img, pos, grid):
    ax = plt.subplot(grid[pos])
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def show_side_by_side(img, saliency_map):
    grid = gridspec.GridSpec(1, 2, wspace=0.)
    render_img_on_grid(img, 0, grid)
    ax = render_img_on_grid(img, 1, grid)
    ax.imshow(saliency_map, alpha=.7)
    plt.show()
    
```

Now we're ready for rapid testing. Here's an image with two classes and its result:

```python

pred, top_5 = make_prediction(bunny_chicks)
saliency_map = compute_saliency_map(model, array=bunny_chicks, target_class=pred)

print(top_5)
show_side_by_side(bunny_chicks_img, saliency_map)

```

[[('n01514859', 'hen', 12.404806),<br>
   ('n02490219', 'marmoset', 12.109007),<br>
   ('n02494079', 'squirrel_monkey', 11.534763),<br>
   ('n02342885', 'hamster', 11.072156),<br>
   ('n02483362', 'gibbon', 10.34959)]]
   
<p align="center">
    <img src="https://slizb.github.io/img/posts/saliency_maps/bunny_chicks.png" width="800">
</p>

Excellent. The model classifies this image as 'hen' and it interestingly focuses most of its attention on just one of the hens. What if the image contained two related objects, where the presence of one was an indication of the presence of the other? Like a can and a can opener:


