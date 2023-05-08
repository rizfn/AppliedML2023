## Dimensionality Reduction

Requirements to do what we've learnt:

* Labelled data (because it's supervised learning)
* Data should lie in the same space: you can't give it a foreign object which it hasn't trained on.

If we have a bunch of data, some is labelled, and we want to label the rest, this is an extremely interesting technique.

But what if we don't have have labels?

## Unsupervised Learning

* We need two identical inputs to provide the same output: if the same input provides multiple outputs, we can't do it (although two different inputs can provide the same output).
* Objects with sufficiently similar inputs should be mapped to similar outputs (other). This is really hard to do in high dimensions, because things are very very far apart. Thus we need to perform some sort of dimensionality reduction, which leads us to point 3:
* We want to map object from the full, $n$ dimensional space with all bands to a smaller one with many neighbours, such that the two previous assuptions continue to hold.

Can we do this? We do have a lot of information, but not all the information is important. We need to find a way for the data to tell us what features are important, without labels.

**PCA:** We have data, we want to change our coordinate axes such that the maximum information falls on some of them. i.e, we're looking to find the optimal projection.

Take the vector that provides the most information, once you project that out, take the next biggest vector, etc etc until you've gotten your required dimension.

That's done by eigendecomposition: We find the eigenvectors and the eigenvalues, and we treat the ones with the highest eigenvalues as the ones which capture the most information.

PCA only looks for linear combinations: a digit 5 can't be thought of as a linear component of '2' and '1'. If you have data that isn't linear, PCA is the wrong option (similar to Fisher's) and thus we should be using some other methods of dimensionality reduction.

## t-Stochastic Neighbour Embedding

t-SNE: You want a nice flat 1D clustering of your data. It's not a projection, like PCA, but just a general clustering.

Put every point in a random position on a line, and then do gradient descent to get to grouping. You want to iterate over points, make a rule by which you move them.

The rule is based on similarity: you move points so they're closer to things they're similar to, and they're repelled by things that are dissimilar.

We can measure the similarity by a distance between 2 points. In that case, we need to choose an appropriate distance metric, such that points in one group are closer to each other but points in another group repel. You can do this by picking an expected distribution (t-SNE is called so because it uses a $t$-distribution, it just works better than a gaussian, because the heavier tails creates more repulsion and thus defines cluster boundaries better) where the 'similarity score' is defined by the height of the distribution at the given distance.

We calculate the distances (and thus the similarities) to every other point. Next, we need to normalize it, so all similarities add up to 1. You normalize it in order to keep low density clusters: otherwise, a single high density cluster will be fine, but other clusters (which have lower density) will not be clustered tightly, despite that being what you want.

t-SNE has a parameter called 'perplexity': It's equal to the expected density. What it'll do is give you a chance to decide if you're interested in primarily high-density environments, or if you're even interested in looking at small clusters of low density as well. You'll never get both, because when you do an embedding from high dimenions to lower, you lose information. Perlexity is about tuning your reducer for high density or low density environments.

Ultimately, in the end, you're calculating the similarities for every pair of points, and so you have a matrix, in the original dimensionality. Then, you can do that for the random 1D case, where you can calculate a new distance matrix. Now, we just have to make the 1D matrix approach the 'ideal' matrix calculated in the full dimensional space.

Different hyperparameters can give you different results: how do you know which one is 'correct'? You can think of it like projections: All of them are 'correct', but none of them tell you the whole story because of the loss of information.

Problems (and solutions) involve:

* t-SNE is slow and scales poorly
* There's a tradeoff between the global an local structure
* No mathematical formulism that shows exactly what's going on

For the speed, Barnes-Hut approximation speeds it up if you're reducing to 2D or 3D

## UMAP

Another clustering algorithm, but in this one you can rigoursly prove that it does work.

In all algorithms, just because something looks like a clean split, it doesn't mean it is one.


