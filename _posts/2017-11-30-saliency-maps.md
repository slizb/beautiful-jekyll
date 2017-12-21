---
layout: post
title: Where the Machines are Looking
subtitle: When to trust a computer vision model.
image: /img/kid_backpack_thumb.png
tags:
  - computer-vision
  - deep-learning
published: true
---

#### *What is this about?*
In this post, I probe the decision-making process of a neural network model, and expose an example of errant model performance in one of the most popular pre-trained networks. 
#### *Why should I care?*
Neural networks are often considered to be black-box models, whose decisions are made behind a mysterioius veil. While their 
decision making process can indeed be complex, its not all hidden to us, there are tools at our disposal to help make sense of it. 
#### *Want to learn more?*
This blog post is awesome, but you'll learn a lot more from a book than from me... If you're interested in learning more about computer vision, check out these books on Amazon:

<a target="_blank"  href="https://www.amazon.com/gp/product/1449316549/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=1449316549&linkCode=as2&tag=bradsliz-20&linkId=2677dd61d3f60f4591bb72ee360ea69c"><img border="0" src="//ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&MarketPlace=US&ASIN=1449316549&ServiceVersion=20070822&ID=AsinImage&WS=1&Format=_SL250_&tag=bradsliz-20" ></a><img src="//ir-na.amazon-adsystem.com/e/ir?t=bradsliz-20&l=am2&o=1&a=1449316549" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />
<a target="_blank"  href="https://www.amazon.com/gp/product/0123869080/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=0123869080&linkCode=as2&tag=bradsliz-20&linkId=57b94134964d2abd4b9a790fe27fb51b"><img border="0" src="//ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&MarketPlace=US&ASIN=0123869080&ServiceVersion=20070822&ID=AsinImage&WS=1&Format=_SL250_&tag=bradsliz-20" ></a><img src="//ir-na.amazon-adsystem.com/e/ir?t=bradsliz-20&l=am2&o=1&a=0123869080" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />
<a target="_blank"  href="https://www.amazon.com/gp/product/1482251884/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=1482251884&linkCode=as2&tag=bradsliz-20&linkId=f6240405db45014bdde0044460806b1f"><img border="0" src="//ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&MarketPlace=US&ASIN=1482251884&ServiceVersion=20070822&ID=AsinImage&WS=1&Format=_SL250_&tag=bradsliz-20" ></a><img src="//ir-na.amazon-adsystem.com/e/ir?t=bradsliz-20&l=am2&o=1&a=1482251884" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />

## Background
One of the biggest critiques of deep learning models is that they have limited interpretability. It turns out that [multiple techniques have been developed](http://cs231n.github.io/understanding-cnn/) specifically to visualize how they make decisions. One such technique is known as saliency mapping, discussed by Karen Simonyan et al. in their paper [*Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps*](https://arxiv.org/pdf/1312.6034.pdf). In their paper, they describe saliency like this:
> Another interpretation of computing the image-specific class saliency using the class score derivative
> is that the magnitude of the derivative indicates which pixels need to be changed the least to affect 
> the class score the most. One can expect that such pixels correspond to the object location in the image. 

In other words, saliency can be computed for a given image, and a given class. **It tells us which pixels in the image contribute most to the model's classification of that class.** Cool!

There happens to be a great python package by Raghavendra Kotikalapudi called [*keras-vis*](https://github.com/raghakot/keras-vis) that supports saliency map visualization for Keras models. Lets try it out!

## Demo
First, we need a model to test out. For simplicity, let's just use a pre-trained model... I choose one of the most popular models out there right now: ResNet:

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
Now, let's pick an image to classify. ResNet was trained on the ILSVRC data (Imagenet), so we need to be sure that the image we choose depicts one of its' classes. I know that some of the classes are cats, so lets try out this cute kitten:

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

def show_side_by_side(img, saliency_map, top_5):
    labels = [x[1] for x in top_5[0]]
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    grid = gridspec.GridSpec(1, 2, wspace=0.)
    render_img_on_grid(img, 0, grid)
    ax = render_img_on_grid(img, 1, grid)
    ax.imshow(saliency_map, alpha=.7)
    ax.text(0.05, 0.05, '\n'.join(labels), 
            transform=ax.transAxes, fontsize=14,
            verticalalignment='bottom', bbox=props)
    ax.text(0.05, 0.05, '\n'.join([labels[0], '', '', '', '']), 
            transform=ax.transAxes, fontsize=14,
            verticalalignment='bottom', color='red')
    plt.show()
    
```

Now we're ready for rapid testing. Here's an image with two classes and its result:

```python

pred, top_5 = make_prediction(bunny_chicks)
saliency_map = compute_saliency_map(model, array=bunny_chicks, target_class=pred)
show_side_by_side(bunny_chicks_img, saliency_map, top_5)

```

<p align="center">
    <img src="https://slizb.github.io/img/posts/saliency_maps/bunny_chicks.png" width="800">
</p>

Excellent. The model classifies this image as 'hen' and it interestingly focuses most of its attention on just one of the hens. 

## Visual Context
What if the image contained two related objects, where the presence of one supports the presence of the other? Like a can and a can opener, or a boy wearing a backpack, or a person playing an instrument? These are examples of visual context clues, and they make for some interesting thought experiments:

<p align="center">
    <img src="https://slizb.github.io/img/posts/saliency_maps/can_opener.png" width="800">
</p>

<p align="center">
    <img src="https://slizb.github.io/img/posts/saliency_maps/boy_backpack.png" width="800">
</p>

<p align="center">
    <img src="https://slizb.github.io/img/posts/saliency_maps/accordion_map.png" width="800">
</p>

I find the model's attention in the can opener image quite interesting. It seems to focus on the can almost as much as it does on the opener! Perhaps this is evidence that **the model has learned this relevant context clue** for the can opener class? Similarly, in the image of the accordion, the model focuses a lot of attention on the person holding the instrument -another valuable context clue, for accordions are to be played, and they cannot be played unless someone holds them!

## Distraction: Revealing potential pitfalls
Next, consider an example where the model does *not* pay attention to the primary object. In these examples, the model pays more attention to other things in the scene than it does to the target class -a basketball:

<p align="center">
    <img src="https://slizb.github.io/img/posts/saliency_maps/dirk.png" width="800">
</p>

<p align="center">
    <img src="https://slizb.github.io/img/posts/saliency_maps/vince.png" width="800">
</p>

The model labels the above images as basketball.  However, instead of focusing on the basketball in the images, the model is mostly looking elsewhere - at the defender, or the arena's background. While the model may again be picking up on important context clues, this is a curious behavior. It suggests that the model is not putting much relative weight on the target object. That makes me suspicious of it's ability to generalize to basketballs in other environments. I also wonder how it would perform given the same enviornment without a basketball. Let's try masking the basketball from the top image, and see what happens:

```python

from scipy.misc import toimage

dirk_no_ball = dirk.copy()
dirk_no_ball[5:45, 35:75] = 0 

```

When we run this through our ResNet model, here's what comes out:

<p align="center">
    <img src="https://slizb.github.io/img/posts/saliency_maps/dirk_no_ball.png" width="800">
</p>

The result is barely different when the basketball is removed! And, amazingly, the model still classifies the image as basketball. This is either an impressive display of the model's ability to pick up on visual context, or a disapointing indication of over-fitting. Let's try one more case, where everything in the image is masked *except* the basketball:

<p align="center">
    <img src="https://slizb.github.io/img/posts/saliency_maps/just_ball.png" width="800">
</p>

Yikes! This should be the easiest example yet, but the model is just way out-of-bounds (*ba-doom-tss*). It appears that somewhere in the training process, the model learned to ignore the basketball in the image, and just look at its surroundings. If I were making an application to detect basketballs, this would not cut it. With this example, we have a perfect case of how saliency maps can help uncover errant model performance.

## Conclusions

Let's recap what I've done here. I introduced saliency maps, which tell us which pixels in an image contribute most to a model’s classification of that image to a given class. Then, I demonstrated how to make a saliency map using the *keras-vis* package, and I used a gaussian filter to smoothe out the results for improved interpretation. Finally, I used some saliency map examples to demonstrate the concepts of visual context learning, and poor generalization. If you're building neural networks for classification, saliency maps can be valuable tools for explaining the decision making process, and debugging unexpected behavior.

That's all!  I hope you enjoyed the post!
