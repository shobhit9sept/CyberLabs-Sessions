# Summary of self supervised learning for image embedding (selfie)

## Existing Problem
In the current time, we have a lot of image data of which very little is labeled and the process of labeling the data by human intervention is fine consuming and extremely laborious. We need methods of utilizing the unlabeled data and train our model using unsupervised learning followed by supervised learning similar to NLP model BERT.

BERT uses feed forward architecture which provides ground for use in image type data, but problem arises due to continuous nature of images compared to distinct words in a sentence.

## Implementation
Similar to BERT we use distractor patches, which mask some parts of the input image and we need a model to predict position of the said distractor patches.

We first pre-train the model on unlabeled data (specifically the first 3 blocks of ResNet-50), call it P (Patch Processing Network).

#### In the encoder
small patches are sent into P and feature vectors are extracted for each patch. The resulting feature vectors are then passed through an attention pooling block to get a vector u and a positional embedding of a distractor patch is also appended, call the output v.

#### In the decoder 
no attention pooling takes place but the vector v for each patch is sent into the computational loss to perform an unsupervised classification task. The encoder and decoder are used jointly to predict the correct patch position.

The encoder learns to compress the information of the image and when positional information of distractor patch is given, it can predict the position of the missing patch accurately. For this the network needs the understanding of the entire image as well as the local context of each patch with respect to its surroundings.

For 224x224 images 32x32 patches are made resulting in a 7x7 grid. With each iteration step, the error decreases, more and more red patches (incorrect) as changes to correct ones. The only places the model faces a challenge is where patches have similar content like for skies, water, trees and so on.

## Why attention pooling?
The benefit of using Attention Pooling over Average or Max pooling is that it can use the self attention mechanism to pick the best set of output dimensions which will end up going into the final affine layer(s). Simple max/average pooling simply squash the 2 dimensional matrix to 1 dimensional but attention ensures that the output dimensions (from 4D to 2D) is best fitted for our task and then only squashing takes place.

## Novelty
#### Positional Embeddings
Positional embeddings of the distractor patches need to be learned. Take an example of image of size 32x32 and patch size of 8x8, we will have 16 resulting patches of which we can learn the embeddings of each patch easily. But for an image of size 224x224 and patch size 32x32, we have total of 7x7=49 patches. So, to reduce the computation, each embedding is decomposed to a row and a column embedding and the resulting embedding is found by summing the components of the same. 

Hence, we will have to learn only 7+7=14 different positional embeddings which greatly reduces the number of parameters and helps in regularization as well, as we are less prone to overfitting now.
