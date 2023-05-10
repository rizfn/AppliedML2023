# Feature Importance

Input feature ranking - SHAP values.

Being able to tell which variables are important is great, but most basic methods only really do it 'on average'. We'd like to do it on a single case: you're sick because of your temperature, etc. One of the few things that went from economics to physics, instead of the other way round.

This is a branch of Interpretable Machine Learning, it's a way to understand why the models do what it does.

You could potentially try all combinations of variables, which isn't good (goes by $N!$ for $N$ parameters). 

For trees, they typically do it by:

1. Weight: How often is a feature usedto split th data across all trees
2. Cover: The weight weighted by the number of training poinst that go through those splits
3. Gain: You add in how much it improves your loss function

## Permutation Importance

A feature is important if shuffling it's values increases the loss function, otherwise it's unimportant. You take a column and shuffle it, and feed that back into your network. You _don't retrain the model_.

First make a model with all the variables. Loop over the features, and generate a matrix where you permute feature j (on the validation set), calculate the error, and see how much the error goes up.

This is a great way to do it, because you only train the model once. Then it's very fast.

## SHAP values

SHAP is a technique for deconstructing an ML's models predictions into a sum of contributions from each of it's input variables. Now, for each individual case, you have an importance for each variable. You can get the overall by summing over it.

Shapley is an economist who won the nobel prize for this. In game theory, he asked how important each player is to the overall cooperation, and how much payoff should each player expect.

He wanted multiple properties in his solution, see the slides:
* **Efficiency:** Summing over agents gives the overall
* **Linear:** More games should just add
* **Null player:** null player gets 0
* **Stand alone test:** If you contribute more, you should always get more, monoticity
* **Anonymity:** Doesn't matter what order people come in
* **Marginalism:** Only uses the marginal contribution of someone

You want a function that takes in a caolition and gives a real value, and an empty gives 0. You calculate it by taking a sum over all correlations, and add a weight. The weight is basically (what is the correlation with you vs the correlation without you).

However, it has a factorial which is bad: but smarter people have found approximations which'll work on ML. Developed by Lundberg, called SHAP (Shapley Additive exPlationations). Despite not looping over the factorial, it is still computationally heavy.

What do you do if PI and Shapley give you different values? Just choose the important values from each importance ranking and predict, and add a ROC curve. You can see which of the 'important features' are more important.

If you have a ton of variables, should you dimensionality reduce or just take the most important features? It depends on your system and fields, honestly. If all your features seem important, a dim. reduction might be better, but if you have some garbage features, just throw them out on the basis of importance.


# Clustering

Together with Dimensionality reduction, it's an Unsupervised algorithm. It's an artform, it doesn't always work out of the box. Like dimensionality reduction, it needs some pre-processing on the data. It's typically when a point is too far away that it ruins the algorithm, so you artificially pull it closer, etc.

Unsupervised algorithms like this are a good way of getting some idea of your data initially.

Evaluating clustering sucks. It's subjective, and typically requires a domain expert. You usually try it on synthetic data and see if your clustering works to begin with. The only decent idea is the elbow method: More clusters will always give you a better score, but if you see a sharp drop and it flattens, that suggests the number of clusters in the data. However, in practise, you don't see stuff like this.

* Heirarchical clustering: Gives clusters a heirarchy: agglomorative and divisive
* Partition clustering: Center based, density based, spectral based


## k-means

Throw $k$ random points, and ask who's close, and iterate. 

Heirarchical clustering: Take every point as it's own cluster, define a 'distance metric', take the two closest clusters and move them

Patitiioning:
* Center based: like k-means, builds clusters around centers
* Density based: clusters around dense regions (DBScan)
* Spectral based: Uses eigenvalues of the similarity patrix to do it before clustering, (PCA+ or something)

k-means: choose 3 random 'centers', and see which points are closest to which ceners. Then, take the centroid of each cluster, and make that the new centroid. It's been improved to k-means++ (better initial guesses) and k-mediods (uses medians instead of means?)

It works very well. Despite starting in a terrible place, you will converge to a nice clustering.

## DBSCAN

Another classic. It won a prize as an algorithm because it's been used for so many things and survived for so long.

Points are either core points, reachable points, and outliers.

A point is a _core point_ if it has $N$ points at a distance of $\epsilon$ from it. It's directly reachable if it's at a distance $\epsilon$ away from any core point, while it's an outlier if it's in neither.

The presence of outliers isn't bad, it allows you to not force points that don't belong to any group to be in one. It gives a different solution to k-means: you can get a circle in a cluster, while you can't in k-means.

## Expectation Maximisation algorithm

Assume data fall in blobs, and throw gaussians on your data. Then find the likelihood, and move your gaussians around (changing mean and sd, in all dimensions) to find the clusters.


Scikit-learn has multiple clustering algorithms: see slides.

