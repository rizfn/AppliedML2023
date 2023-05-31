# Generative Adversarial Networks (GANs)

## Variantional Autoencoders

Autoencoders: You have an input, boil it down to a small latent space, and reconstruct the input. However, there are complications. Autoencoders are from the 80s, but things didn't explode at the time because of a lack of computational power. Called the AI Winter.

The latent space of an autoencoder is extremely complex, because it's not regularlized. Even if you're close in the latent space, you're not necessarily similar in the original space. You can't just take a 'random number' in this space, because even if it's close it'll spit out complete nonsense. Thus, you can't use it to generate new stuff.

The answer is Variational Autoencoders. Now, the encoder output parameters have to come from a pre-defined definition in the latent space for every input. For example, you set it to be a multi-dimensional Gaussian. You encode it into numbers, but now they can't be any numbers, but they'll have to be Gaussianly distributed. It's less efficient than a normal AE though.

We start by encoding and decoding, and we make the latent space caussian by using a smart Loss function. We have a 'reconstruction loss' which guarantees that you only encode in terms of the means and the variances. You add that to the 'similarity loss', which is typically KL divergence.

'Reconstruction loss' asks if the two are alike, 'similarity loss' asks if the latent space looks Gaussian, apparently. You might also want to scale the similarity loss, instead of a normal addition, to 'force it' to do it.

With this, one can even do arithmetic in latent space. Similar to words: king - man + woman = queen. Now, the latent space becomes 'nice', 'continuous', and 'useful'.

## GANs

You train a variational autoencoder to encode data to a latent space. Now, you can just feed in random noise in the latent space into the decoder, which you train. It doesn't work immediately, but you train another network (called a 'Discriminator'), which has to decide if the image it gets is real or fake.

Simply take the images, take a decoder that generates images from noise, and a discriminator that identifies whether the generator's images are real or not. The loss function is just how easily you can differentiate them.

Eventually, the generator will be able to generate stuff that looks like the original thing. The loss function is a (scaled) sum of the typical loss plus an adversarial loss, and the adversarial loss is measured from the ability of the discriminator to tell if it's real or fake.

We tell networks what to do by enforcing additional parts on the loss function.

## Reinforcement Learning

Machine learning paradigms typically fall under 'supervised' and 'unsupervised' learning. However, Reinforcement Learning is normally thought of as a third paradigm.

You get no data at all in reinforcement learning, you only get rules. You can define rules as a markov decision process (MDP). The actual model doesn't need to know the decision process, it just needs to know a score. Say, in the MDP, if you move from one state to antoher you gain score, if you move from one to another, you lose score.


# Nvidea GPUs

* Scaling up: On the same system.
* Scale out: Utilizing multiple systems

If you know what you have to do, you can hardcore the operations onto the hardware: that's what started the idea of GPUs. Typically, you'd want to do transformations and stuff on pixels.

Because game developers wanted something more flexible, GPUs developed to be able to do flexible stuff, but still in parallel. Instead of doing a hardcoded operation, you do something you choose (but, say, for every pixel). The term used was GPGPU (General Purpose GPUs).

To make it easier to use the GPU for general purpose calculations, Nvidea developed CUDA (OpenCL is a standardized one that also supports AMD). CPUs and GPUs have very different design ideas. ALUs are what  perform the calculations, you have control logic (which is stuff that makes your code run faster on hardware, stuff like re-ordering instructions, trying to predict when an application takes memory, etc). It take sup a lot of transitors, and even more are taken up by the L2 cache, which stores memory.

Single thread speedups in recent years aren't due to the ALU (as clock speeds aren't really increasing) but because of optimizations to the control logic and the L2 cache.

A GPU is much more parallelism focussed: it has way more ALUs but much less control logic. It also has less cache. The programmer has to make sure that the data provided fits the architecture, but you're rewarded with much greater raw power.

[The APIs developed by RAPIDs:](https://docs.rapids.ai/api)

The basic python library to use the GPU's power is `cuDF`. You can do a lot of operations on dataframes automatically with it. It's a drop in replacement of pandas, with a lot of features.

`KviklO` is an API to access disk memory: you can read from disk directly to GPU. If the hardware supports GDS (GPU direct service?). The APIs in C++ and python are written to feel natural.

`cuML` is for machine learning, covering basically scikitlearn. You can combine it with `cuDF` like how you'd combine sklearn and pandas.

`libcudf` powers the python libraries, it's a c++ backend that supports the same operations, you can use it for other languages like rust too.

`cuxfilter` is a visualization library that allows you to load raw data without preprocessing directly into hte browser.

More slides about parallelizing out: see slides. Mainly dask.






