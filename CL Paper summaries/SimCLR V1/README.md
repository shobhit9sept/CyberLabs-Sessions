# An overview of the SimCLR framework
The major components of the SimCLR framework include:

1) Data Augmentation
2) A Base Encoder f(x)
3) A Projection Head g(h)
4) The Contrastive Loss Function.

## Data Augmentation
The SimCLR framework starts by fetching examples of images from an original dataset. It transforms the given image example into two corresponding views of the same example image.

While previous methods to contrastive learning introduced architecture changes, SimCLR argues that a target image’s random cropping sets up enough context for contrastive learning. The use of cropping enables the network to learn the global to local contrast and contrast the same image’s adjacent views.

The paper also mentions a systematic study performed, that combined the different compositions of data augmentations—for example, combining cropping with other data augmentation techniques such as blur, color distortion, and noise. 

## A base encoder f(x)
The base encoder f(x)
 uses a Convolutional Neural Network (CNN) variant based on the ResNet architecture. It extracts image representation vectors from the augmented data images produced by the data augmentation module. This extraction produces the embeddings, h
.

## A projection head g(h)
The projection head g(h)
 consists of two fully-connected layers, i.e., a multi-layer perceptron (MLP), that takes in the embeddings, h
, as its inputs from the base encoder and produces an embedding z
. This module’s role is to map the image representations to a latent space where contrastive loss is applied.

The contrastive loss function is a modified version of the cross-entropy loss function, which is the most widely used loss function for supervised learning of deep classification models. The function is shown below.

## The contrastive loss function

(IMG)

The contrastive loss function states that the similarity of zi
, and zj
 corresponding to, for example, an image of a cat and its augmentation should be closer together. In other words, they should attract.

In contrast, the similarity of any k
, which is not i
, should be pushed further apart (repel). An example of this would be the representation of a dog, and a cat should repel eachother.

That’s a simplistic view of what the contrastive loss function does in a nutshell.
