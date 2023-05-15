# Convolutional Neural Network

For a 100x100 image, just using an NN is very bad, and there are much smarter ways to do it.

There are more modalities than tabular data: images, sequences, point coulds and graphs, etc. You can combine them to get most other things: video = image+sequence, robotics = tabular+sequence/graph, etc.

We want to understand each modality individually, because it's easy to mix them once you do. Today we're focussing on images, mainly Computer Vision. You can have classification (is it a cat), detection (draw a bounding box around it) or segmentation (divide the exact subjects in the image).

Images are typically represented with each pixel as a vector: len 1 for greyscale, length 3/4 for colour images. We typically normalize to get numbers between 0 and 1.

We can invent a filter/kernel, which is a hypothesis about a feature we think is interesting. In reality, it's a matrix of the same size as the 'neighbourhood' of the feature. We then scan the kernel across the image, multiply it and sum the matrix together (called "mapping" the kernel). For every position, we then get a value: we take the _maximum_ value among them. That way, we only need the feature to be matched in one location.

This is called _Max Pooling_, because we're bringing down a 256x256 image to a single number (which is the 'score').

A neural network will learn the filter on it's own, instead of having us give it to them. Currently, everything is 'flat', because each pixel is multiplied by a learnable parameter. However, we can add more depth to the model.

Instead of a 2x2 matrix as the kernel, we can have a 4x4 tensor. It allows for some way of finding more 'abstract' features.

Why don't we use a basic, feed-forward NN to do it? We can prove that it can, with infinite data, but a CNN is better because it considers translational symmetry: The location of where an image is shouldn't change the result, which makes it much less complex. The FFNN has no sense of 'translational invariance'.

Note that we've added translational invariance into a CNN by design: that's an _inductive bias_. We can do another: we know that there's a heirarchy of valeus: Low level features like edges, which high level features like faces.

We can handle that by stacking the features, whith each filter applied to the output of the previous convolution. We stack multiple encoders to learn features. You typically do convolution+Relu, and follow it up with Pooling.

## Convolutional Arithmetic:

* **Kernel Size:** The kernel size must be smaller than your image size, which would reduce the size of your output
* **Padding:** In order to have the output the same size as the input, we pad the borders with dummy pixels. They can be zeros. It also helps the kernel capture behaviours at the edge.
* **Stride:** Pixel close to each other are often related, so instead of scanning over every pixel, we can skip a few of them. That's called 'striding'.

## Architecture:

After all the convolution and pooling, you can then just use a simple FFNN, because it works perfectly. It just doesn't work well on the input, but once your NN has abstracted the data it's much easier.

## Segmentation:

What if we want to detect objects, instead of just classifying? We want to label every pixel, as to whether it belongs to the objecet or not. It requires returning at output of the samesize as the input image. We do a 'transposed convolution', to upsample our results.

This is semantic segmentation, where you classify every pixel. Instance segmentation, on the other hand, has a notion of 'multiple objects', and (I think??) instead of being 'true/false' per pixel it can capture different values for if it's part of something.

The Unet goes all the way down to a single pixel of a very large latent space. That's called the bottleneck: it's used to capture information of the whole image.

## Generational models:

You can create new stuff, GAN (generative adversarial networks).

## Transfer learning with feature extraction:

If you train it on cats/dogs, the initial section (the features learnt from the convolution and stuff) is usable for almost everything: it's just the final bit (the decoder) that needs to be redone. Features like 'dots' and 'edges' are present everywhere.

Thus, you should always start with a pre-trained network: to quote daniel, "it's a free lunch".







