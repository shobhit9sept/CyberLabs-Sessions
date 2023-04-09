
# DenseNet Summary

The paper proposes a new neural network architecture called DenseNet that enables better training and accuracy in image classification tasks by using some new techniques.

## Problems faced earlier -

Deep neural networks suffer from the vanishing gradient problem, where the gradients become too small as it reaches the beginning or the end of the model. This problem leads to degradation in training accuracy as the depth of the network increases.

## Key ideas proposed in the paper - 

(IMG)

The authors propose a new architecture that addresses the vanishing gradient problem and improves training and test accuracy.

1) DenseNet introduces "skip connections" that allow the earlier layers to receive direct input from later layers. These connections help combat the vanishing gradient problem and make it easier for the network to learn complex features.
2) DenseNet also introduces "feature reuse", where each layer receives inputs from all previous layers. This improves the flow of information and reduces the number of parameters required.
3) To ensure maximum information flow between layers in the network, we connect all layers (with matching feature-map sizes) directly with each other. To preserve the feed-forward nature, each layer obtains additional inputs from all preceding layers and passes on its own feature-maps to all subsequent layers.
4) DenseNet layers are very narrow (e.g., 12 filters per layer), adding only a small set of feature-maps to the “collective knowledge” of the network and keep the remaining feature maps unchanged—and the final classifier makes a decision based on all feature-maps in the network.
5) One big advantage of DenseNets is their improved flow of information and gradients throughout the network, which makes them easy to train. Each layer has direct access to the gradients from the loss function and the original input signal, this helps training of deeper network architectures. Further, we also observe that dense connections have a regularizing effect, which reduces overfitting on tasks with smaller training set sizes.




## Architecture of DenseNet - 

(IMG)

1) Consider a single image x0 that is passed through the network. The network comprises L layers, each of which implements a non-linear transformation Hl(·), where l indexes the layer. Hl(·) can be a composite function of operations such as Batch Normalization (BN), rectified linear units (ReLU), Pooling [19], or Convolution (Conv). We denote the output of the l th layer as xl.

2) The l th layer receives the feature-maps of all preceding layers, x0, . . . , x`−1, as input: xl = Hl([x0, x1, . . . , xl−1]), referring to the concatenation of the feature-maps produced in layers 0, . . . , l−1. Because of its dense connectivity we refer to this network architecture as Dense Convolutional Network (DenseNet). For ease of implementation, we concatenate the multiple inputs of Hl(·) into a single tensor.

3) To facilitate down-sampling in our architecture we divide the network into multiple densely connected dense blocks.We refer to layers between blocks as transition layers, which do convolution and pooling.

4) If each function Hl produces k featuremaps, it follows that the l th layer has k0 +k ×(l−1) input feature-maps, where k0 is the number of channels in the input layer. An important difference between DenseNet and existing network architectures is that DenseNet can have very narrow layers, e.g., k = 12. We refer to the hyperparameter k as the growth rate of the network. The growth rate regulates how much new information each layer contributes to the global state.

5) Before entering the first dense block, a convolution with 16 output channels is performed on the input images. For convolutional layers with kernel size 3×3, each side of the inputs is zero-padded by one pixel to keep the feature-map size fixed. We use 1×1 convolution followed by 2×2 average pooling as transition layers between two contiguous dense blocks. At the end of the last dense block, a global average pooling is performed and then a softmax classifier is attached. The feature-map sizes in the three dense blocks are 32× 32, 16×16, and 8×8, respectively.


## Training and Testing -

Networks are trained using stochastic gradient descent (SGD). On CIFAR and SVHN we train using batch size 64 for 300 and 40 epochs, respectively. The initial learning rate is set to 0.1, and is divided by 10 at 50% and 75% of the total number of training epochs. On ImageNet, we train models for 90 epochs with a batch size of 256. The learning rate is set to 0.1 initially, and lowered by 10 times at epoch 30 and 60. 


(IMG)

## Datasets used -

1) CIFAR - Natural images with 32×32 pixels. CIFAR-10 (C10) consists of images drawn from 10 and CIFAR-100 (C100) from 100 classes. The training and test sets contain 50,000 and 10,000 images respectively, and we hold out 5,000 training images as a validation set.

2) Image Net - 1.2 million images for training, and 50,000 for validation, from 1000 classes.

3) SVNH - The Street View House Numbers (SVHN) dataset [24] contains 32×32 colored digit images. There are 73,257 images in the training set, 26,032 images in the test set, and 531,131 images for additional training.


## Overfitting - 
One positive side-effect of the more efficient use of parameters is a tendency of DenseNets to be less prone to overfitting. We observe that on the datasets without data augmentation, the improvements of DenseNet architectures over prior work are particularly pronounced.


## Conclusion - 

DenseNets tend to yield consistent improvement in accuracy with growing number of parameters, without any signs of performance degradation or overfitting.

They require substantially fewer parameters and less computation to achieve state-of-the-art performances.

DenseNets naturally integrate the properties of identity mappings, deep supervision, and diversified depth. They allow feature reuse throughout the networks and can consequently learn more compact and, according to our experiments, more accurate models. Because of their compact internal representations and reduced feature redundancy, DenseNets may be good feature extractors for various computer vision tasks that build on convolutional features.

