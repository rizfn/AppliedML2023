# Lecture 1

The only day where we won't directly do ML.

## Introduction to Machine Learning

A nice 'intuitive' definition of machine learning is

> Machine learning programs can perform tasks without being explicitly programmed to do so.

## Universal Approximation Theorem

It's a bit like epsilon-delta, which defines what it's like to be differentiable. It says that with a Neural network type function, you can approximate any other function, given you have enough degrees of freedom. 

In reality, we don't satisfy it (cause we have categorical data, etc). It just says that *there exists* a function, part of this course involves actually finding it.

To get it, we often assume that there's no noise in the training set, and while you have infinitely many functions that pass through the training point, they won't have the same error o nthe unseen point.

## Stochastic Gradient Descent

It's a general way used to get parameters and weights of an ML algorithm. Instead of doing the steepest approach and running over all data, you do it over a small portiona of data, and take steps. 

We knows that there is a solution and it can be found, but we need to develop an algorithm. There are 2 general ways, trees and Neural Networks (NNs). Trees are fast and performant, but can't be generalized, while NNs are the opposite.

## Dimensionality and Complexity

Humans are bad at high-dimensional problems, even linear ones, while computers are good at it (linear algebra and calculus together do the trick). However, we're typically good at non-linear problems, as long as they're in low-dimensions.

## Types of ML

Either Supervised or Unsupervised ML. 

* **Supervised:** you already know the answers, and you can train and validate on that. Either classification or regression. 
* **Unsupervised:** you don't know that. You can either cluster things, or reduce the dimensionality.

We'll mostly be covering supervised learning.

Machine learning typically allows for better discrimination of null and alternate hypotheses: i.e, it creates good classifiers. How do we measure separation? If it's a boolean (healthy vs sick), you can use the confusion matrix terms (false positives, false negatives, true positives, true negatives). If it's a score, you use a **ROC curve**.

Typically, we'll be using ROC curves to quantify whether your ML algorithm is good. For a more detailed explanation, see the lecture in applied statistics that covers them.

Classifying things linearly can be done very easily with **Fisher's Discriminant** (see appstat lecture). If non-linear, you want to use trees. Typically, you want multiple trees and combine them to get a better answer. How do you get multiple trees from the same data? One way is boosted decision trees, where you 'boost' the points it's got wrong, and remake a new tree (see advanced appstat lecture)

Neural Networks use a ton of linear stuff, because it's fast. They use an 'activation function' which is non-linear, to combine the signals. It can be any sigmoid function.

Dimensionality and how well methods work: For things that fit in an excel sheet, trees are the off-the shelf think to use. For Kernel methods (like k-NN, nearest neighbours), it depends on dimensionality. Essentially, you don't want outliers: you want your neighbours to be all around you. If you have a wall next to you, you're an outlier. In high dimensions, most things are on the boundary, and so it's pretty bad.

You won't expect your ROC curve to be perfect, it's always limited by how much information you have in your data.


