# Hyperparameter Optimazation


## Different Methods

* Naieve manual approah
* grid search, 
* random search, 
* bayesian optimization

### Naive

You basically do trial and error: go over some values with a certain step. Also called babysitting or Grad Student Descent. Training and testing takes a while, and so it's challenging.


### Grid search

Curse of dimensionality. More parameters makes stuff harder. It basically takes the cartesian product, and is a raster scan. More systematic than before, but still not ideal.


### Random Search

Here, if the minimum is in between grid points, you will still find it. If one parameter is irrelevant, it doesn't matter when taking time to run (for some reason). This scales better for higher dimensions than grid search. It has almost no assumptions about the algorithm, it only needs a range/distribution. You manually specify the distribution, and it samples random hyperparameters from that that it finds the result.


### Bayesian Optimization

If it takes a long time to train, think a little before training the next batch of hyperparameters. We need a probabilistic surrogate model and an acquisition function.

You make a model for how the loss function looks with your given points, along with it's uncertainity. The aquisition model asks "where is it probable that a maximum is"? You then run it for that point, update the surrogate model, and once again pick the point with the highest acquisition value. The time taken to do the calculation is nothing compared to training.

Initially, it does nothing: it needs like 20 random points to actually choose new points smartly. However, in the long run, it makes a massive difference. You choose points based on where you'd expect good values.

The acquisition function is made in such a way that you shouldn't get stuck in a local minimum too often. If you only get values in one place, the acquisition values far away might increase.

Comes with python package `bayes_opt`.

In low dimensions, you could potentially do a full search over every combination.

### Bayesian Optimization with Hyperband (BOHB):

Start with multiple, and throw out the bad seeds. A bit like mixing in a particle swarm. Python package `optuna`.


## General Tips:

Not all hyperparameters matter equally: for NNs, the *learning rate* is the most important parameter typically.

Stuff like XGBoost and LightGBM work well because they use rules of thumbs for hyperparameters given your input data. Coded in with if statements, typically. 




