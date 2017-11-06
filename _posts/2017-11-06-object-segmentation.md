---
layout: post
title: Object Detection and Segmentation in Python with Mask-RCNN
subtitle: Visual examples and discussion on the uses of object detection and object segmentation with Mask R-CNN.
image: /img/federer_thumb.jpeg
tags: [computer-vision, deep-learning]
---


#### *What is this about?*
In this post, I walk through some hands-on examples of object detection and object segmentation using Mask R-CNN. 
#### *Why should I care?*
Besides being super cool, object segmentation can be an incredibly useful tool in a computer vision pipeline. Say you are training a CV model to recognize features in cars. If you have images of cars to train on, they probably contain a lot of background noise (other cars, people, snow, clouds, etc.). Object detection / segmentation can help you identify the object in your image that matters, so you can guide the attention of your model during training.

#### *Want to learn more?*
This blog post is awesome, but you'll learn a lot more from a book than from me... If you're interested in learning more about object detection and segmentation, check out these books:

<a target="_blank"  href="https://www.amazon.com/gp/product/0470976373/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=0470976373&linkCode=as2&tag=bradsliz-20&linkId=25dee7f7984f191284a8bf3d11b84e29"><img border="0" src="//ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&MarketPlace=US&ASIN=0470976373&ServiceVersion=20070822&ID=AsinImage&WS=1&Format=_SL250_&tag=bradsliz-20" ></a><img src="//ir-na.amazon-adsystem.com/e/ir?t=bradsliz-20&l=am2&o=1&a=0470976373" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />
<a target="_blank"  href="https://www.amazon.com/gp/product/9811051518/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=9811051518&linkCode=as2&tag=bradsliz-20&linkId=8ec9a09862cad06d8172d58b81bbe022"><img border="0" src="//ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&MarketPlace=US&ASIN=9811051518&ServiceVersion=20070822&ID=AsinImage&WS=1&Format=_SL250_&tag=bradsliz-20" ></a><img src="//ir-na.amazon-adsystem.com/e/ir?t=bradsliz-20&l=am2&o=1&a=9811051518" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />
<a target="_blank"  href="https://www.amazon.com/gp/product/331952481X/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=331952481X&linkCode=as2&tag=bradsliz-20&linkId=937957f3ca82af92d2d07f122b6abf13"><img border="0" src="//ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&MarketPlace=US&ASIN=331952481X&ServiceVersion=20070822&ID=AsinImage&WS=1&Format=_SL250_&tag=bradsliz-20" ></a><img src="//ir-na.amazon-adsystem.com/e/ir?t=bradsliz-20&l=am2&o=1&a=331952481X" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />
<a target="_blank"  href="https://www.amazon.com/gp/product/168083116X/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=168083116X&linkCode=as2&tag=bradsliz-20&linkId=6b91189d8b370a6de27a67c4e556dc88"><img border="0" src="//ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&MarketPlace=US&ASIN=168083116X&ServiceVersion=20070822&ID=AsinImage&WS=1&Format=_SL250_&tag=bradsliz-20" ></a><img src="//ir-na.amazon-adsystem.com/e/ir?t=bradsliz-20&l=am2&o=1&a=168083116X" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />

## Background
Earlier this year, Kaiming He et al. released [their paper *Mask R-CNN* on arXiv](https://arxiv.org/abs/1703.06870). (If you're familiar with computer vision or deep learning, you may recognize Kaiming's name from another recent contribution -[Resnet](https://arxiv.org/abs/1512.03385) ). In the *Mask R-CNN* paper, they make some impressive claims, including superior performance on a number of object detection and segmentation tasks. Here is their abstract:
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

Just this past week, the codebase for *Mask R-CNN* was [made public on GitHub](https://github.com/matterport/Mask_RCNN), and it generated [a bit of buzz on Reddit](https://www.reddit.com/r/MachineLearning/comments/7a7hm1/p_tensorflow_implementation_of_mask_rcnn_for/). Since I was in need of a pre-trained object detection model for work, I decided to take it for a test drive, and check out its segmentation performance too.

## Installation
Installation was not exactly simple, but also not not *too* complicated. I started with the instructions on the GitHub page, but found I needed a bit more. I went through the steps generally as follows:
1. clone the Mask_RCNN repository
2. download the pre-trained COCO weights (mask_rcnn_coco.h5) from the repository's releases page
3. move the weights to the just created `Mask_RCNN` directory
4. upgrade tensorflow to >= 1.3
5. clone the [special forked version of COCO](https://github.com/waleedka/coco)
6. run make under `coco/PythonAPI`
7. run python setup.py under `coco/PythonAPI`
8. install missing dependencies using `conda` / `pip` until setup works
9. try to run `demo.ipynb` from jupyter and cross fingers

## Demo
The demo.ipynb notebook that comes with the Mask R-CNN repository is really very good.  I'll use that as a starting point.

After importing packages and modules, loading the pre-trained model and setting up initial parameters, we get right into the meat of the demo. First, note the object classes that are defined:

```python

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
               
```
 
 This object class list effectively defines the limits of the pre-trained model.  Make sure your class is in the list. If it is, you're good to go with no more training. Otherwise, you'll have to re-train the model with examples of your class.

Now let's take a look at some model ouputs. The demo comes with some pre-canned images that are all impressive, but I'm always skeptical, so I'm going to test with my own image. The tennis racket object class caught my eye, so I'll grab a random tennis image from google. And here it is... Roger Federer in action during a tennis match:

<p align="center">
    <img src="https://raw.githubusercontent.com/slizb/slizb.github.io/master/img/federer_raw.png" width="600">
</p>

Running the image through the pretrained model is simple:

```python

# Load a the image
image = scipy.misc.imread('federer.jpeg')

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'])
                            
```

And here is the output:

<p align="center">
    <img src="https://raw.githubusercontent.com/slizb/slizb.github.io/master/img/federer.png" width="670">
</p>

Beautiful!  The bounding boxes are accurate, and the segmentation masks are just stunning.  The model accurately bounds the racket, the ball, and Federer himself, and the masks seem to be nearly spot on... even following the boundaries of Roger's flowing hair. The only mistakes seem to be clipping off the ends of his fingers. 

Though I only passed one image to the model, it is possible to pass a batch of images as elements of a list. Thus, when I inspect the results, I only need to look at the first element.  It contains a dictionary with the following elements: 

* **"rois"**: image coordinates of the object bounding boxes
* **"class_ids"**: class ids for the detected objects
* **"scores"**: softmax confidence associated with each object class assignment
* **"masks"**: a binary array indicating the segmentation boundary for each object on the image

#### *A thought before moving on*:
Though *Mask R-CNN* seems to work really well out of the box, it is also pretty slow.  It took about 8 seconds to score a single image on my Mac, while consuming all 8 CPU cores. So, if speed is important to you, you may consider alternatives. [*YOLO*](https://pjreddie.com/darknet/yolo/) is one option that can perform object detection in real time:

<iframe width="854" height="480" src="https://www.youtube.com/embed/VOC3huqHrss" frameborder="0" gesture="media" allowfullscreen></iframe>

That said, *Mask R-CNN* seems to perform just fine for my usecase. Let's try another example. For my project, I need to detect cars. How will it perform if there are multiple cars to segment? Here's an image with a lot of cars:

<p align="center">
    <img src="https://raw.githubusercontent.com/slizb/slizb.github.io/master/img/many_cars.jpg" width="600">
</p>

And Here's the labeled ouptut image from *Mask R-CNN*:

<p align="center">
    <img src="https://raw.githubusercontent.com/slizb/slizb.github.io/master/img/many_segmented_cars.png" width="640">
</p>

Again, the performance is impressive. There doesn't seem to be any problem with multiple objects. In my work, though, the image usually has a bit more background noise, and the objects are not so distinctly separated. Most of the time, one or more objects are occluded. Lets try an example like that:

<p align="center">
    <img src="https://raw.githubusercontent.com/slizb/slizb.github.io/master/img/cows_and_cars.jpg" width="600">
</p>

Here's the output:

<p align="center">
    <img src="https://raw.githubusercontent.com/slizb/slizb.github.io/master/img/segmented_cows_and_cars.png" width="645">
</p>

The model appropriately juggles this confusing scene. Cows are cows, cars are cars, people are people. Even people inside of cars are identified. Despite all of the background noise and object occlusion, the performance is still pretty excellent. There are some subtle mistakes, like the truck boundary bleeding onto the bus, and a couple cows being identified as two seperate objects, but some of these mistakes may even be acceptable for a human to make.

Let's try one more example.  In my work, there is always a primary object -a car, which may or may not be surrounded by other objects and noise (other cars, people, snow, clouds, etc.). Here's a typical case:

<p align="center">
    <img src="https://raw.githubusercontent.com/slizb/slizb.github.io/master/img/car_lot.jpg" width="600">
</p>

And the model output:

<p align="center">
    <img src="https://raw.githubusercontent.com/slizb/slizb.github.io/master/img/segmented_car_lot.png" width="640">
</p>

One assumption we can make in this scenario is that the target car is also the largest identified object. Likewise, we can apply a little logic to isolate it. In simple terms, we just measure the area of each box, and grab the biggest one.  Here's some code for doing just that:

```python

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_width(xy):
    width = abs(xy[1] - xy[3])
    return width

def get_height(xy):
    height = abs(xy[0] - xy[2])
    return height

def get_area(xy):
    width = get_width(xy)
    height = get_height(xy)
    area = width * height
    return area

def get_biggest_box(xy_list):
    biggest_area = 0
    for i, xy in enumerate(xy_list):
        area = get_area(xy)
        if area > biggest_area:
            biggest_area = area
            biggest_xy = xy
            ix = i
    return biggest_xy, ix

def overlay_box(image, xy): 
    position = (xy[1], xy[0])
    width = get_width(xy)
    height = get_height(xy)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    rect = patches.Rectangle(position, 
                             width, 
                             height,
                             linewidth=1,
                             edgecolor='r',
                             facecolor='none')
    ax.add_patch(rect)
    plt.show()
    
```

If we plug the results of our previous example into this workflow, it looks like this:

```python

big_box, big_ix = get_biggest_box(r['rois'])
overlay_box(image, big_box)

```

<p align="center">
    <img src="https://raw.githubusercontent.com/slizb/slizb.github.io/master/img/big_car_boxed.png" width="640">
</p>

Great! We've isolated the object we care about. Let's take it a step further, and black out everything outside of our box:

```python

def make_box_mask(image, xy):    
    target = image[xy[0]:xy[2], xy[1]:xy[3], :]
    img = np.zeros_like(image)
    img[xy[0]:xy[2], xy[1]:xy[3], :] = target
    plt.imshow(img)
    
make_box_mask(image, big_box)

```

<p align="center">
    <img src="https://raw.githubusercontent.com/slizb/slizb.github.io/master/img/big_car_masked.jpg" width="640">
</p>

Even Better! By doing this, we can limit the noise in our image. **This is really important, and can be exploited to imporve a variety of computer vision tasks.** For example, If we apply such a technique to all of the training images in a machine learning task that needs to focus on a target object, our neural network won't even be tempted to focus on anything but the target objects. 

As you may have guessed, though, we can extend this technique further.  Let's apply the same technique using the segmentation mask output from *Mask R-CNN* for even finer results:

```python

def make_segmentation_mask(image, mask):
    img = image.copy()
    img[:,:,0] *= mask
    img[:,:,1] *= mask
    img[:,:,2] *= mask
    plt.imshow(img)

mask = r['masks'][:,:,big_ix]
make_segmentation_mask(image, mask)

```

<p align="center">
    <img src="https://raw.githubusercontent.com/slizb/slizb.github.io/master/img/big_car_seg_masked.png" width="640">
</p>

Wow!  Now we're left with virtually nothing but the target car object in the image. No more background noise, no more distracting objects. And its all built into a highly automatable pipeline. You may be worried about the segmentation boundaries cutting off pieces of your object, which is a valid concern. You could always add a buffer to the edges, but I'll leave that to  you to figure out. 

## Conclusions
Let's wrap up what I've done here: 
I installed *Mask R-CNN*, and tested it on a variety of images including some with a lot of noise and object occlusion. The model consistently impressed me with its performance, though it was pretty slow. Then, I demonstrated how a pre-trained object detection model like *Mask R-CNN* could be used to screen training data for machine learning tasks that need to focus on a primary object.



That's all, I hope you enjoyed reading!




