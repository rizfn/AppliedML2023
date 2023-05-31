# Graph Neural Networks (GNNs)

Motivation for it: in CNNs, we take a kernel and take a convolution across our image. Whatever value that we get out will include locality, because the value depends on the neighbourhood. The distance between the neighbouring cells is constant. Sometime called 'unstructured' data, sometimes 'superstructured' by DIKU guys.

What if our data is not an image, but we want to use a CNN? For example, we have some mathematical discrete graph. We want to use a CNN because they're really really good, as they incorporate locality in our data.

If we use a CNN on this, we're forcing irregular geometry into images: we're shaping the problem to the tool, and not the tool to the problem. Instead, we use a specialize tool call a **Graph Neural Network**

### Graphs

A graph is a set of nodes and edges. Types include:
* Undirected
* Directed
* Weighted (with distances on edges)
* Signed
* Multigraph (different types of connections)

Unlike images, graphs have no underlying assumptions on the geometry of the data: that's provided by the user, in the form of the edges.

### Graph Convolutions

Similar to a CNN: it's just not a fixed number of pixels, but you convolve over all 'neighbours' of a node. Your output is then another graph of (possibly) different dimensionality, analogous to images. Many different types of convolution, for example, edgeconv:

$$ \tilde{x_j} = \sum^n_{i=1} f(x_j, \;x_j - x_i) $$

$\tilde{x}$ represents an updated node.

We loop over the neighbours, and we need a learnable function $f$. The solution (cause we don't konw the function) is to use an NN, and train it (eg: [Multi-layer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)). If we know that the function is simple, we use a linear approach.

This is the start, but we just keep doing it like a CNN, and then eventually we boil it down into numbers (by flattening?) and make it a feed-forward NN.

After concatenation, you typically want to put it through a shallow but wide NN to give it more degrees of freedom to 'shuffle'. You use node aggregation to make something work on any number of nodes: for example, take the mean, min, max and sum.

Transformers can be thought of as graphs on steriods, becasue they take the locality into account while also keeping track of _attention_.






