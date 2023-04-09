
# Fast R-CNN 

## Introduction:

Fast R-CNN is a method for object detection that improves on previous models.

Object detection is a complicated task because it requires accurate localization of objects.

Fast R-CNN uses a single-stage training algorithm to jointly classify object proposals and refine their spatial locations.

Previous models like R-CNN and SPPnets had drawbacks such as being expensive in space and time for training and slow for object detection.

Fast R-CNN is faster and more accurate than previous models because it uses several innovations to improve training and testing speed.

The model jointly learns to classify object proposals and refine their spatial locations, which makes it more efficient.

Fast R-CNN is based on deep convolutional networks, which have been successful in image classification problems.

## Why Fast R-CNN?

We propose a new training algorithm that fixes the disadvantages of R-CNN and SPPnet while improving their speed and accuracy. We call this method Fast R-CNN because it’s comparatively fast to train and test. The Fast RCNN method has several advantages:
Higher detection quality (mAP) than R-CNN, SPPnet.
  Training is single-stage, using a multi-task loss.
  Training can update all network layers.
  No disk storage is required for feature caching.

## Architecture:

This network takes as input an entire input image and a set of object proposals. Then for each object proposal, a region of interest(RoI) pooling layer extracts a fixed-length feature vector from the feature map. Each feature vector is fed into a sequence of fully connected layers that finally branch into two sibling output layers:
One that produces softmax probability estimates over K object classes plus a catch-all “background” class,
Another that gives output four real-valued numbers for each of the K object classes. 



## RoI pooling layer:

The RoI pooling layer converts features inside a rectangular region of interest (RoI) into a fixed size feature map.

The RoI is defined by a four-tuple (r, c, h, w) that specifies its top-left corner and its height and width.

The RoI pooling layer divides the RoI window into an H × W grid of sub-windows.

Each sub-window has an approximate size of h/H × w/W.

The values in each sub-window are max-pooled into the corresponding output grid cell.

The hyper-parameters H and W are independent of any particular RoI.

## Initialization from pre-trained networks:
When a pre-trained network initializes a Fast R-CNN network, it undergoes three transformations. 

First, the last max pooling layer is replaced by a RoI pooling layer that is configured by setting H and W to be compatible with the net’s first fully connected layer (e.g., H = W = 7 for VGG16). 

Second, the network’s last fully connected layer and softmax (which were trained for 1000-way ImageNet classification) are replaced with the two sibling layers described earlier (a fully connected layer and softmax over K + 1 categories and category-specific bounding-box regressors). 

Third, the network is modified to take two data inputs: a list of images and a list of RoIs in those images.

## Fine-tuning for detection:
We propose a more efficient training method that takes advantage of feature sharing during training. In Fast RCNN training, stochastic gradient descent (SGD) mini batches are sampled hierarchically, first by sampling N images and then by sampling R/N RoIs from each image. Critically, RoIs from the same image share computation and memory in the forward and backward passes.

Fast R-CNN uses a streamlined training process with one fine-tuning stage that jointly optimizes a softmax classifier and bounding-box regressors rather than training a softmax classifier, SVMs, and regressors in three separate steps.

## Multi-task loss: 
Fast R-CNN network has two sibling output layers. The first outputs a discrete probability distribution (per RoI), p =(p0, . . . , pK), over K + 1 categories. As usual, p is computed by a softmax over the K+1 outputs of a fully connected layer. The second sibling layer results in bounding-box regression offsets, tk = (tkx , tky , tkw , tkh ) for each of the K object classes, indexed by k.
  
Where:
  Lcls(p, u) = − log pu is log loss for true class u. 


u - a ground-truth class.
v - a ground-truth bounding-box regression target.
The Iverson bracket indicator function [u ≥ 1] evaluates to 1 when u ≥ 1 and 0 otherwise.


## Back-propagation through RoI pooling layers:
Let xi ∈ R be the i-th activation input into the RoI pooling layer and let yrj be the layer’s j-th output from the rth RoI. The RoI pooling layer computes yrj = xi ∗(r,j), in which i∗ (r, j) = argmaxi‘∈R(r,j) xi’. R(r, j) is the index set of inputs in the sub-window over which the output unit yrj max pools. A single xi may be assigned to several different outputs yrj. The RoI pooling layer’s backward function computes the partial derivative of the loss function with respect to each input variable xi by following the arg max switches:


## Results:
#### VOC 2010 and 2012 results: 
Fast R-CNN achieves the top result on VOC12 with a mAP of 65.7% (and 68.4% with extra data).
On VOC10, it achieved a result with a mAP of 66.1%.

#### VOC 2007 results:
The improvement of Fast R-CNN over SPPnet illustrates that even though Fast R-CNN uses single-scale training and testing, fine-tuning the convolutional layers greatly improves mAP (from 63.1% to 66.9%).

#### Training and testing time:
                            Runtime comparison between the Fast RCNN, R-CNN, SPPnets.


#### Truncated SVD:
Truncated SVD can reduce detection time by more than 30% with only a small (0.3 percentage point) drop in mAP without needing additional fine-tuning after model compression.


In this technique, a layer parameterized by the u × v weight matrix W is approximately factorized as
                                                W ≈ UΣtVT
using SVD. Truncated SVD reduces the parameter count from u*v to t*(u + v), which can be significant if t is much smaller than min(u, v). 

#### Which layers to fine-tune? 
For the less deep networks considered in the SPPnet paper, fine-tuning only the fully connected layers appeared to be sufficient for good accuracy.
 To validate that fine-tuning the convolutional layers is important for VGG16, we use Fast R-CNN to fine-tune but freeze the thirteen convolutional layers so that only the fully connected layers learn. This experiment verifies our hypothesis: training through the RoI pooling layer is important for very deep nets.

#### Does multi-task training help? 
Multi-task training is convenient because it avoids managing a pipeline of sequentially-trained tasks. But it also has the potential to improve results because the tasks influence each other through a shared representation (the ConvNet).

Across all three networks, we observe that multi-task training improves pure classification accuracy relative to training for classification alone.

#### Do SVMs outperform softmax? 
Fast R-CNN uses the softmax classifier learned during fine-tuning instead of training one-vs-rest linear SVMs post-hoc, as was done in R-CNN and SPPnet.

This effect is small, but it demonstrates that “one-shot” fine-tuning is sufficient compared to previous multi-stage training approaches.

#### Are more proposals always better?
Using selective search’s quality mode, we sweep from 1k to 10k proposals per image, each time re-training and retesting model M. If proposals serve a purely computational role, increasing the number of proposals per image should not harm mAP.

We find that mAP rises and then falls slightly as the proposal count increases (Fig. 3, solid blue line).The state-of-the-art for measuring object proposal quality is Average Recall (AR). AR correlates well with mAP for several proposal methods using R-CNN, when using a fixed number of proposals per image. 

#### Preliminary MS COCO result:
We applied Fast R-CNN (with VGG16) to the MS COCO dataset to establish a preliminary baseline. The PASCAL-style mAP is 35.9%; the new COCO-style AP, which also averages over IoU thresholds, is 19.7%.

## Conclusion:
This paper proposes Fast R-CNN, a clean and fast update to R-CNN and SPPnet. Of particular note, sparse object proposals appear to improve detector quality.




