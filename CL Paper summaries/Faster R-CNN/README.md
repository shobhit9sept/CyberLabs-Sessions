# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks



## Approach:
Our object detection system, called Faster R-CNN, is composed of two modules. The first module is a deep fully convolutional network that proposes regions, and the second module is the Fast R-CNN detector that uses the proposed regions. The entire system is a single, unified network for object detection. Using the recently popular terminology of neural networks with ‘attention’ mechanisms, the RPN module tells the Fast R-CNN module where to look.

The RPN module is responsible for generating region proposals. It applies the concept of attention in neural networks, so it guides the Fast R-CNN detection module to where to look for objects in the image.
                                            



## Novelties:

#### Region Proposal Networks
1) We introduce novel Region Proposal Networks (RPNs) that share convolutional layers with state-of-the-art object detection networks. By sharing convolutions at test-time, the marginal cost for computing proposals is small (e.g., 10ms per image).
2) The convolutional feature maps used by region-based detectors, like Fast RCNN, can also be used for generating region proposals. On top of these convolutional features, we construct an RPN by adding a few additional convolutional layers that simultaneously regress region bounds and objectness scores at each location on a regular grid. The RPN implements the terminology of neural network with attention to tell the object detection (Fast R-CNN) where to look.
3) The RPN does not take extra time to produce the proposals compared to the algorithms like Selective Search. Due to sharing the same convolutional layers, the RPN and the Fast R-CNN can be merged/unified into a single network. Thus, training is done only once.

The RPN works on the output feature map returned from the last convolutional layer shared with the Fast R-CNN.  Based on a rectangular window of size nxn, a sliding window passes through the feature map. For each window, several candidate region proposals are generated. 

#### Anchors
In contrast to prevalent methods that use use pyramids of reference boxes in the regression functions or pyramids of filters we introduce novel “anchor” boxes that serve as references at multiple scales and aspect ratios.
1) Our scheme can be thought of as a pyramid of regression references which avoids enumerating images or filters of multiple scales or aspect ratios. This model performs well when trained and tested using single-scale images and thus benefits running speed. An anchor is centered at the sliding window in question, and is associated with a scale and aspect ratio. 
Generally, there are 3 scales and 3 aspect ratios and thus there is a total of K=9 anchor boxes. But K may be different than 9. In other words, K regions are produced from each region proposal, where each of the K regions varies in either the scale or the aspect ratio. 

2) Translation-Invariant Anchors -
An important property of our approach is that it is translation invariant, both in terms of the anchors and the functions that compute proposals relative to the anchors. If one translates an object in an image, the proposal should translate and the same function should be able to predict the proposal in either location.

3) Multi-Scale Anchors as Regression References -
Our anchor-based method is built on a pyramid of anchors, which is more cost-efficient. Our method classifies and regresses bounding boxes with reference to anchor boxes of multiple scales and aspect ratios.

              (IMG)                  

## Feature Sharing between RPN and Fast R-CNN
The 2 modules in the Fast R-CNN architecture, namely the RPN and Fast R-CNN, are independent networks. Each of them can be trained separately. In contrast, for Faster R-CNN it is possible to build a unified network in which the RPN and Fast R-CNN are trained at once.

We discuss three ways for training networks with features shared:

(i) Alternating training. In this solution, we first train RPN, and use the proposals to train Fast R-CNN. The network tuned by Fast R-CNN is then used to initialize RPN, and this process is iterated. This is the solution that is used in all experiments in this paper.

(ii) Approximate joint training. In this solution, the RPN and Fast R-CNN networks are merged into one network during training . In each SGD iteration, the forward pass generates region proposals which are treated just like fixed, pre-computed proposals when training a Fast R-CNN detector. The backward propagation takes place as usual, where for the shared layers the backward propagated signals from both the RPN loss and the Fast R-CNN loss are combined. But this solution ignores the derivative w.r.t. the proposal boxes’ coordinates that are also network responses,so is approximate.

(iii) Non-approximate joint training. The RoI pooling layer in Fast R-CNN accepts the convolutional features and also the predicted bounding boxes as input, so a theoretically valid backpropagation solver should also involve gradients w.r.t. the box coordinates. These gradients are ignored in the above approximate joint training. In a non-approximate joint training solution, we need an RoI pooling layer that is differentiable w.r.t. the box coordinates. 


## Loss Function
For training RPNs, we assign a binary class label (of being an object or not) to each anchor. We assign a positive label to two kinds of anchors: (i) the anchor/anchors with the highest Intersection-over- Union (IoU) overlap with a ground-truth box, or (ii) an anchor that has an IoU overlap higher than 0.7 with any ground-truth box.       

             (IMG)



## Combining all of the above: 
For each nxn region proposal, a feature vector (of length 256 for ZF net and 512 for the VGG-16 net) is extracted. This vector is then fed to 2 sibling fully-connected layers:
The first FC layer is named cls, and represents a binary classifier that generates the objectness score for each region proposal (i.e. whether the region contains an object, or is part of the background).
The second FC layer is named reg which returns a 4-D vector defining the bounding box of the region.
The first FC layer (i.e. binary classifier) has 2 outputs. The first is for classifying the region as a background, and the second is for classifying the region as an object.

For training the RPN, each anchor is given a positive or negative objectness score based on the Intersection-over-Union (IoU).


                        (IMG)



#### Difficulties:
One drawback of Faster R-CNN is that the RPN is trained where all anchors in the mini-batch, of size 256, are extracted from a single image. Because all samples from a single image may be correlated (i.e. their features are similar), the network may take a lot of time until reaching convergence.

## Conclusion:
We have presented RPNs for efficient and accurate region proposal generation. By sharing convolutional features with the down-stream detection network, the region proposal step is nearly cost-free. Our method enables a unified, deep-learning-based object detection system to run at near real-time frame rates. The learned RPN also improves region proposal quality and thus the overall object detection accuracy.


