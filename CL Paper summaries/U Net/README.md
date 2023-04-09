
# U-NET: Convolutional Networks for Biomedical Image Segmentation
## Objective of the paper:
Successful training of deep neural networks requires many thousand annotated 
training samples. We can’t get a lot of data for every problem, especially in 
medical tasks. This paper presents a network and training strategy that relies on 
the strong use of data augmentation to use the available annotated samples more 
efficiently.
The typical use of convolutional networks is on classification tasks, where the 
output to an image is a single class label. However, in many visual tasks, 
especially in biomedical image processing, the desired output should 
include localization i.e., a class label is supposed to be assigned to each pixel. 
## Idea and Architecture:
The U-Net architecture is built upon the Fully Convolutional Network and 
modified in a way that it yields better segmentation in medical imaging. The 
paper use’s excessive data augmentation by applying elastic deformations to the 
available training images. This allows the network to learn invariance to such 
deformations, without the need to see these transformations in the annotated 
image data.
The U-Net comprises of two parts an encoder/contraction path and a 
decoder/expansion path. The contraction path consists of a repeated application 
of a 3x3 convolutions(unpadded) each followed by a ReLU and a 2x2 max 
pooling operation with stride 2 for downsampling. At each downsampling step, 
we double the number of feature channels. This captures context through a 
compact feature map.

                        (IMG)


The expansion path consists of upsampling of the feature map followed by a 2x2 
convolution(“up-convolution”) that halves the number of feature channels a 
concatenation with the cropped feature map from the contracting path, and a 3x3 
convolution, followed by a ReLU. The upsampling of the feature dimension is 
done to meet the same size as the block to be concatenated on the left. The 
expansion helps in getting more features but loses the localization, the 
localization information is concatenated from the contraction path.
The cropping is necessary due to the loss of border pixels in every convolution. 
At the final layer, a 1x1 convolution is used to map each 64-components feature 
vector to the desired number of classes. The new ideas introduced in this paper 
are:
#### 1. Overlap- tile strategy
Prediction of the segmentation in the yellow area requires image data within the 
blue area as input. To predict the pixels in the border region of the image, the 
missing context is extrapolated by mirroring the input image. This tiling 
strategy is important to apply the network to large images, since otherwise the 
resolution would be limited by the GPU memory.
#### 2. Data augmentation
Along with the usual shift, rotation, and color adjustments, they added elastic 
deformations. This was done with a coarse (3x3) grid of random displacements, 
with bicubic per-pixel displacements. This allows the network to learn 
invariance to such deformations, without the need to see these transformations in 
the annotated image corpus. This is important in biomedical segmentation since 
deformation is the most common variation in tissue and realistic deformations 
can be simulated efficiently.
#### 3. Separation of touching objects of the same class
This is done using a weighted loss, where the separating background labels 
between touching cells obtain a large weight in the loss function. This forces the 
network to learn the small separation borders between touching cells. The loss 
function included per-pixel weights both to balance overall class frequencies and 
to draw a clear separation between objects of the same class. The basic idea of 
per-pixel weighting is to add a class weight (to upweight rarer classes), plus 
“morphological operations” — find the distance to the two closest objects of 
interest and upweight when the distances are small. This encourages the network 
to learn to draw pixel boundaries between objects.
## Conclusion:
This paper focuses on utilizing a U-shaped network for the task of image 
segmentation specifically for biomedical images. Since, biomedical images 
aren’t easily available, the authors use various data augmentation techniques to 
artificially increase their dataset in addition to modifying their structure to 
support limited training samples and increase context as well as using a weighted 
loss function to improve performance. Though, this paper focuses on biomedical 
images, this work can be applied in various field such as quality control and 
manufacturing. Their framework could also be extended to other domains

