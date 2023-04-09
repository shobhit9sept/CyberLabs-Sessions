
# The Transformer – Attention is all you need


Recurrent neural networks (RNN), long short-term memory networks(LSTM) and gated RNNs are the popularly approaches used for Sequence Modelling tasks such as machine translation and language modeling. However, RNN/CNN handle sequences word-by-word in a sequential fashion. This sequentiality is an obstacle toward parallelization of the process.

Moreover, when such sequences are too long, the model is prone to forgetting the content of distant positions in sequence or mix it with following positions’ content.

Attention mechanisms are one of the solutions to overcome the problem of model forgetting. This is because they allow dependency modelling without considering their distance in the input or output sequences. Due to this feature, they have become an integral part of sequence modeling and transduction models. 

However, in most cases attention mechanisms are used in conjunction with a recurrent network.

## Architecture

Neural sequence transduction models generally have an encoder-decoder structure. The encoder maps an input sequence of symbol representations to a sequence of continuous representations. The decoder then generates an output sequence of symbols, one element at a time.

![attention](link)

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.

The authors are motivated to use self-attention because of three criteria.  

1) One is that the total computational complexity per layer.
2) Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.
3) The third is the path length between long-range dependencies in the network.
The Transformer uses two different types of attention functions:

#### Scaled Dot-Product Attention 
computes the attention function on a set of queries simultaneously, packed together into a matrix.

#### Multi-head attention
allows the model to jointly attend to information from different representation subspaces at different positions.

A self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires O(n) sequential operations.

In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length is smaller than the representation dimensionality, which is often the case with machine translations.

## Key Takeaways
This work introduces Transformer, a novel sequence transduction model based entirely on attention mechanism.

It replaces the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers for translation tasks.

On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, the model achieves a new state of the art.  In the former task the model outperforms all previously reported ensembles.
