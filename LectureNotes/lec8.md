# RNNs and NLP

Words are things that come in time-series, because they're connected in a certain order. They also don't have a fixed length. You constantly have the question of 'how far back' do you have to remember.

## Recurrant Neural Networks

Usual feed-forward NNs will build in a straight (horizontal) line. In an RNN, you add in the time data orthogonally. You have your input from the current time step, but you also have a 'hidden state', which contains information from the past. You process it in a simple way (matrix multiplication + non-linear function) and then you get the hidden state for the next step. You hope that the hidden state has information from the past: not just the immediate closest step, but even before then.

To predict the values, you can either use the outputs or you can use the hidden state to predict some global properties of the time-series. You can use the same concepts of max-pooling from CNNs.

### Training RNNs

You need to deal with the hidden states: the method is **Back propagation through time** (eg: LSTM). You start from the end, compute the gradients of the unit, and the go back and re-compute the gradients of the same parameters. You then take the average of all those gradients, and then use that to update the unit.

There's a problem with long-term dependencies. The gradients vanish, as you propagate in the network, which is called Vanishing Gradients. It's not a problem specific to RNNs, it even occurs in Deep NNs. The deep layers typically have very small gradients, and they solve them by adding "residual connections" or "skip connections". For us, however, the NN is deep but in a non-trivial way because of the time-series data. This problem can be addressed with LSTM, Long Short Term Memory.

Here, you have something more complicate to calculate the hidden state. Apart from the hidden state, you have a *cell state* which takes care of it, composed of *gates*. There's a *forget gate*: Multiplies the cell-state by 1s and 0s, depending on what you want to forget. Crucially, in the cell state, you add the information calculated by the forget gate, because in that case the gradient doesn't shrink (multiplication does shrink it).

You can also simplify if with GRU: instead of multiple output gates, input gates, etc  you just have an update gate. You have 1 hidden state, but you just have an addition in the update. If you don't work with NLP, and work with just numbers, it could be as powerful as LSTM.

### Back to RNNs

You can make an RNN multilayered (by stacking them) and you can also make them bidirectional. In a bidirectional RNN, you have information travelling backwards in time too, with LSTM.

Exploding gradients: A problem not sepcific to RNNs. Your loss landscae  may have cliffs, and so gradients can experiece explosive growth, even with a small learning rate. Gradient clipping (setting a max value to the gradient) is one possible fix. Adding a 'warm up' in training is the standard practise: start with a very small learning rate and gradually increase it.

### Practical tips:

* Different sequence lengths? You want to zero pad it.
* Pack/mask padding inputs to increase performance, because quite often, for many time-steps your NN could be doing nothing if you have tons of zeros. `tf.keras.layers.Masking`
* Use a random seed, or start from an old (pre-trained) model
* Gradient clipping
* Initial distribution matters for LSTM

## NLP

How do we represent words? We can just enumerate every word, but it's more efficient in practise to take 'parts of words' instead of entire words, caled BPE: Byte Pair Encoding. Called 'tokenizing': splitting words and text into a series of numbers.

You can embed all the words in a multidimensional space. Each word is assumed to comprise of 'latent factors', which sum up to the meaning of the word.

### Language Modelling

You need to predict the next word in a sequence. A language model is one which predicts the next word given the previous ones in the sequence. Until 2017, LSTMs dominated the field, but since then Transformers are now the dominant algorithms (a paper called *attention is all you need* from google).

### Transformers:

For every input, you compute 3 vectors by multiplying by three matriced, $Q, K, V$ for all inputs (the same ways as RNN store weights). For every input, you take pairs of them, take the scalar product, normalize by the dimension of keys, and take the softmax. You need to do this for every pair of inputs. You then multiple the softmax by $V$, which gives you the attention, which is passed to an MLP.

You take the inputs, embed into a linear space, apply the attention, but to the outputs you add the inputs (skip connection). It's in two parts: an encoder and an decoder, for translation. For GPT models, they only use a decoder because the previous tokens can't know the earlier words. 

* It's extremely parallelizable. 
* Typically trained on fixed sequence lengths. 
* The complexity grows quadratically with the sequence length, because you need to cpmpute all pairs of $Q$ and $K$, unlike an RNN where it's linear.

Transformers typically require a lot of data and are less straightforward to train.

Transformers typically use **positional encoding:** take the fourier transform of the positions in the time series, and add that to the vector embedding.


# Autoencoders

An encodere generates a smaller version of the signal, like a summary. An autoencoder contains an encoder and a decoder.

The encoder will represent some inputs in a 'latent space', and we train a decoder to do the inverse of the encoding. It's _unsupervised_ learning: you want to train it to encode and decode so that it gets the same thing out. There's something smart in it, either in the latent space, or in the actual encoding. Used a lot with images.

A PCA is a linear Autoencoder.

Used for
* Unsupervised learning (eg: clustering) on images, sound, graphs, etc
* Compression (with loss!) of images
* De-noising and inpaiting imagees
* Anomaly detection
* Training on large datasets with few labels

The most important hyperparameters are
* Size of the latent space
* Architecture of the NN
* Loss function

### De-Noising Autoencoders

Say you have some noise that ruins your autoencoder, and you want to remove that noise at the end of the decoder. You do this by artificially adding in noise, and training your decoder to reproduce the original (un-noisly) data. In that way, you can de-noise numbers.

This idea can be extended to reconstructing larger parts of an image, called "inpainting". See the slides: you can draw around, and thus 'influence' the autoencoder to predict what you believe.

### Anomaly detection

You spit out a lot of stuff, and need to do some sort of quality assuarance. You can just take pictures of it. One problem with a factory is if you're checking for anomalies, you might not have a lot of them, and thus you don't know what an anomaly looks like. There's a too small sample size of anomalies for supervised learning.

Instead, you can tran an AE on all images, they'd be predominantly good. Then, ask what's the difference between the original data and the reconstruction? If the input is something it knows, the error will be small. However, if the input is an anomaly, the filters aren't there in the encoder, and so you'll have a large error.

Of course, in this case you want a small latent space, so the autoencoder should not be 'general'.

### Training assistance

Say you have a large data set but only a small subset are labelled. You can take the large data set, make an autoencoder, and look at the encoded data in latent space. You can now make a small network classifier (or regressor) which uses them as inputs, and train it on the small subset of labelled data.



