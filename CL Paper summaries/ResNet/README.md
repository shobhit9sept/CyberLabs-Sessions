
# ResNet Paper Summary 

This is the code and summary of the famous research paper ResNet.
  


## Idea and inspiration
By that time researchers believed that "the deeper the better" when we talk about convolutional neural networks. And it even made sense since models were then more capable of learning more complex data. However they noticed that after some depth, the performance decreses. This was the main issue with VGG that they couldn’t go as deeper as they wanted because they encountered overfitting!!

Since neural networks are good function approximators, they should be able to easily solve the identify function, where the output of a function becomes the input itself.

                     f(x) = x

Following the same logic, if we bypass the input to the first layer of the model to be the output of the last layer of the model, the network should be able to predict whatever function it was learning before with the input added to it.

                    f(x) + x = h(x)

The intuition is that learning f(x) = 0 has to be easy for the network.                    
## Challenges faced by these architectures 
One of the problems ResNets solve is the famous known vanishing gradient. This is because when the network is too deep, the gradients from where the loss function is calculated easily shrink to zero after several applications of the chain rule. This result on the weights never updating its values and therefore, no learning is being performed.

With ResNets, the gradients can flow directly through the skip connections backwards from later layers to initial filters.
## Architecture
![Model architecture](https://github.com/shobhit9sept/CyberLabs-Sessions/blob/main/CL%20Paper%20summaries/ResNet/Images/Architecture.jpg)

Each of the layers follow the same pattern. They perform 3x3 convolution with a fixed feature map dimension (F) [64, 128, 256, 512] respectively, bypassing the input every 2 convolutions. Furthermore, the width (W) and height (H) dimensions remain constant during the entire layer.

The dotted line is there, precisely because there has been a change in the dimension of the input volume (of course a reduction because of the convolution). Note that this reduction between layers is achieved by an increase on the stride, from 1 to 2, at the first convolution of each layer; instead of by a pooling operation, which we are used to see as down samplers.

![Table](add link)

![3Darchitecture](add link)

Every layer of a ResNet is composed of several blocks. This is because when ResNets go deeper, they normally do it by increasing the number of operations within a block, but the number of total layers remains the same — 4. An operation here refers to a convolution, a batch normalization and a ReLU activation to an input, except the last operation of a block, that does not have the ReLU.

## Results
