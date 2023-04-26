# Machine Learning: Loss Function, Training, Preprocessing

In terms of supervised learning, you either do
* Classification (predicting a boolean)
* Regression (predicting a continuous values)

There are two main types of decision trees:
* Boosted decision trees
* Random Forest

There are more, but we're only covering two here.

If you have an image, or sound, or continuous data coming in that can't fit in a spreadsheet, trees aren't great, but NNs do a good job.

## 10 points about data

1. Check input data (plot it, print first and last 10, check nulls)
2. Split the data into train, valid, test sets to not overtrain
3. Consider features and omit those that don't contribute
4. Ensure lots of useful quality data (either by having it or getting it)
5. Fast computing and access to data (start with smaller data and optimize your code, then run on the entire dataset)
6. If data is in shortage, consider methods for augmenting it
7. All data is flawed, make sure you know how and filter accordingly
8. Decide whether to impute or filter missing data
9. Test several methods, as different methods apply to different cases
10. Make cross-checks: especially because ML algorithms are hard to look into, and it's hard to find out if something is a bug

## 10 points about algorithms

1. Use appropriate model/architecture that matches the data/problem. If it fits in an excel sheet, use a boosted decision tree. Potentially throw an unsupervised algorithm on it immediately after reading the data
2. Think carefully about the loss function (what do you want to optimize?)
3. Trees are good for getting fast results on structured (excel) data
4. NNs are more performant and versatile but harder to train
5. Variable transformations are typically required for NNs
6. Image analysis uses CNNs (Convolutional Neural Networks)
7. Dimensionality reduction benefits very high dimensional problems.
8. Streams of data (such as text) can be analyzed with LSTM (Long Short Term Memory) /RNN (Recurrant NNs) networks
9. Unsupervised learning / clustering can be difficult to interpret
10. Uncertainities in regression can be given by ML (typically NN) algorithms

## Loss Functions

Classification:

* Zero-one: If you're right, you get a 1, otherwise 0.

We would like the algorithm to come up with a continuous score instead?
* Logistic (log loss), is used quite a lot. Has roots in information theory.
* Focal Loss (weighing one (the 1s or the 0s) more heavily than the other, primarily built for cases where you have very few of one type of event in your training data)
* KL-Divergence: Used a lot in theory
* Exponential
* Hinge (straight line)

Log Loss is also called the Binary cross Entropy.
$$ \mathcal{L} = -\frac1N \sum_{n=1}^N \left[ \,y_n \ln(\hat{y}_n) + (1-y_n) \ln(1-\hat{y}_n) \,\right] $$
$\hat{y}_n$ is your predicted value, while $y_n$ is the true value ($y_n \in [0, 1]$). Depending on the true value, only one term in the sum will contribution.

You can also havev multiclassification, where you give each class a score, and the scores you give sum up to 1

If the data is unbalanced (or skewed, you have very few of one event), you can use AUC (area under the curve) instead. You can also use the F1 score, which is made with similar things (false positives, negaives).

You can also use focal loss, where you put in an exponent into the terms in the sum, and then the algorithm _feels_ it much stronger

For regression, you can use the mean square error (quadratic loss). 
$$\left( \frac{P-T}{T} \right)^2$$
$P$ is predicted, $T$ is true. It focuses on difficult cases, because they go as a square. You could also go by the Mean Absolute Error, which is less sensitive to outliers. 
$$\left| \frac{P-T}{T} \right|$$
It's not differentiable, so Huber made something like it which is continuous in the middle. Finally, people found log-cosh, which has the same properties and is now an industry standard (it's the "default" to use, if you know nothing about your data).

log-cosh is twice differentiable everywhere, while Huber is only once-differentiable.

If you, as an external person, want to prevent your model from doing something, you can add penalty terms to the loss function. For example, XGBoost punishes complexity.

## Stochastic Grandient Descent

When you take a step in the gradient direction, the length of that step is called the _learning rate_. Loss landscapes may be complicated, with many local minima. Good initial conditions are important, remember them.

Stochastic gradient descent has 2 advantages to normal gradient descent.

1. You get some stochasticity in it, it can help you leave local minima
2. It's much faster: looping over the entire dataset takes a long time.

It also vectorizes well. Loss vs iteration will jitter, unlike normal descent.

Too low of a learning rate means slow convergence, while a high learning rate jumps around and never gets there. Crank it up until it fails, and take something just lower than that.

Learning rate schedules: you tweak it with epochs. You start high, and then decay, typically.

## Splitting data

You subdivide your dataset into training, validation, and test. Typically 60-20-20, or 80-10-10. The training set is always the largest, and is submitted to the stochastic gradient descent.

You evaluate on your validation set every epoch. The error on the training set will eventually go to zero, while the error on the validation set reaches a minimum and eventually increases (overtraining). Typically, you choose the point where the validation loss is the minimum. You have a 'patience', which is how much you let the validation set increase before terminating.

Once you've found a good model, you can try it on the test set.

The splitting is the _last_ thing you do before the training: you want to impute, etc beforehand. Also good to randomize, in case there's a trend.

### k-fold Cross Validation

You divide your data into k different folds. You train your classifier on  k-1 folds, and test it with the remaining fold, k-times. Now you have k different models, each trained 'correctly' but on different sets. It's very advisable if your data is small, but if your data is large CV won't really help and takes a lot more time. It allows you to get really nice error bars on your model.

For time series: CV is only done on data from the past, not from the future! Train a little, test on the future of that. Add that future to the training set, test on the future of that, etc.

## Decision Trees

Great out-of-the-box solution, susceptible to overtraining and requires a lot of data. However, simple decision trees are fairly inspectable. However, trees can only spit out a single value, and can't estimate with some sort of regression like an NN can.

A simple decision tree is not a great approximation, a better one is by making multiple trees and combining them. In a boosted decision tree, you 'boost' incorrectly classified data and remake a tree. At the end of the training, trees are combined into an ensemble.

Random forests are a different approach to generate different trees. The trees made in BDTs are correlated: if you combine correlated things, they effect each other and the overall performance doesn't go as $\sqrt{N}$ as you'd expect. Random forests are independent.

If you have a dataset, you pick up random entries from the data set, and sample from it with replacement. In doing so, the trees become a lot more independent. Furthermore, it's parallelizable, unlike BDTs.

