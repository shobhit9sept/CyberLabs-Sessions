
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
1) U-Net architecture is designed for better segmentation in medical imaging.
2) Excessive data augmentation is used by applying elastic deformations to training images.
3) U-Net has two parts: encoder/contraction path and decoder/expansion path.
4) Contraction path involves repeated 3x3 convolutions, ReLU activation, and max pooling with stride 2 for downsampling.
5) Feature channels are doubled at each downsampling step to capture context in a compact feature map.

                (IMG)
                
6) Expansion path involves upsampling of feature map, 2x2 "up-convolution" to halve feature channels, concatenation with cropped feature map from contraction path, 3x3 convolution, and ReLU activation.
7) Localization information is concatenated from contraction path to prevent loss of localization in expansion.
8) Final layer uses a 1x1 convolution to map each 64-component feature vector to desired number of classes.
The new ideas introduced in this paper are:
#### 1. Overlap- tile strategy
Prediction of the segmentation in the yellow area requires image data within the 
blue area as input. To predict the pixels in the border region of the image, the 
missing context is extrapolated by mirroring the input image. This tiling 
strategy is important to apply the network to large images, since otherwise the 
resolution would be limited by the GPU memory.
#### 2. Data augmentation
Along with the usual shift, rotation, and color adjustments, they added elastic 
deformations.

This was done with a coarse (3x3) grid of random displacements, 
with bicubic per-pixel displacements. This allows the network to learn 
invariance to such deformations, without the need to see these transformations in 
the annotated image corpus. 

This is important in biomedical segmentation since 
deformation is the most common variation in tissue and realistic deformations 
can be simulated efficiently.
#### 3. Separation of touching objects of the same class
1) Weighted loss is used to improve performance in object segmentation by forcing the network to learn small separation borders between touching objects.
2) Per-pixel weights are included to balance class frequencies and draw a clear separation between objects of the same class.
3) Class weights are added to upweight rarer classes.
4) Morphological operations are used to find the distance to the two closest objects of interest and upweight when the distances are small.
5) This encourages the network to learn to draw pixel boundaries between objects.
## Conclusion:
This paper focuses on utilizing a U-shaped network for the task of image 
segmentation specifically for biomedical images. Since, biomedical images 
aren’t easily available, the authors use various data augmentation techniques to 
artificially increase their dataset in addition to modifying their structure to 
support limited training samples and increase context as well as using a weighted 
loss function to improve performance. Though, this paper focuses on biomedical 
images, this work can be applied in various field such as quality control and 
manufacturing. Their framework could also be extended to other domains

