# Morning

No notes, initial project and job opportnities. Check slides.

# Afternoon

## RNNs

RRNs are universal: if you use rational numbers, you can prove they're turing complete. However, that's a blessing and a curse: being able to do anything means you can't deduce it. Anything you can say about a turing complete system is either trivial or uncomputable (? some quote, find it).

In practise though, we use finite representations like floats which aren't turing complete but still too strong to reliably train them.

RRNs are trained with Back Propagation Through Time (BPTT). It unrolls an NN which gives a static but deep network. Two problems:
* Optimizing is resourcing intensive
* Fails completely (vanishign gradients, bifurcations in state space)

### State space bifurcation

Think of the simplest RNN, one node that feeds into itself. The behaviour of the system depends on the weight and bias, and for different cases you get different results.

Positive weight, and positive bias (<1) is a fixed point attractor.
Postive >1 weight and positivev bias is a repeller system (but it is a fixed point).
Negative weight <1 leads to oscillatory behaviour. You can have oscillatory attractiveness or oscillatory repulsiveness.

Bifurcations are places where you have an attractor solution (which you can find by optimizing). Plots (slide trouble with RNNs 6/7) are the fiexd point on the y axis, with weights and biases on x. 

With RNNs, you can into points wehre the solution becomes double values, ie, the attractor splits into two. The resulting huge gradients can destroy a lot of learning in a single iteration.

## Echo State Networks (ESNs)

An ESN is weaker than an RNN, but it's simple, fast to train, and stable. There's nothing to converge: the solution is deterministic. You don't optimize, you just solve.

In practise, ESNs end up being stronger than LSTMs, because even though it's got less expressive power if you can't converge you're messed up. More expressive power means that you can predict a large class of behaviours, i.e how complicated functions can you capture within the framework.

The trouble with RNNs is the 'R' (recurrance). However, that's also what gives expressiveness, so we want to not get rid fo it but rather control it. The general idea is to make something that has enough teeth instead of training the recurrant part. The general form is you take an nput, map it into a recurrant resevoir (which serves as a system with memory) and then map it from there into output. The mappings are, in general, matrices or linear maps.

We train to find a way to make the resevoir static, by opimizing the maps.

What makes it work? The hidden matrix $W$ must be sparse and connected.
* Sparsity gives locality: thinsg have a "neighbourhood", and so info can gradually propagate
* Connectedness: information will eventually reach everywhere
* Sparsity also helps making matrix-vector multiplication $O(n)$ over $O(n^2)$

$W$ is a randomly connected matrix (?).

The eigenvalues of W can be thought of the things hich control how memory attenuates. If you act on it with a random vector, the eigenvectors give you the paths through which it travels.

The eigenvalues need to be scaled: W must have a desired spectral radius (largest eigenvalue) $\rho(W) \approx 1$. Too large and it explodes, too small and it dies out.

If you have a random matri with normalized etries in the complex plane, you get the eigenvalues sitting in the unit(?) circle. As the matrix gets larger and larger, it converges to the disc.

### Performance

Instead of gradient descent, training is a single deterministic least-squares calculation. As we only train the output.

$$W^{out} X \simeq D$$

$X$ is input history, $D$ is the data that we train on. This is just a least squares problem. You can also add regularization. It's so fast that you can do online training: it's when you do training on the fly, so you can retry, improve, and then go on.

The Mackey-Glass sequence  is what's used at times to predict chaotic systems. It's chaotic, but still has a lot of predictability. Training online actually gives you an order of magnitude better prediction.

ESNs only work for one dimension, you might need to work with time series of multiple different variables (and thus higher dimensions). Can be dealt with by exploiting the structure in the input matrix: instead of taking a random input matrix, you say it's alike a 3D image. THe input matrix needs to take stuff from the input space to a higher dim hidden space, so they took one that means something. 

They use a mix of pixel valeus, convolutions, discrete cosine transforms, gradients, and some random stuff for good measure.

## Anomaly detection

You need to know how well it predicts, but you often can't go pixel by pixel because it'll be huge, for chaotic systems. You can quantify normal behaviour based on "how well you predict something".

Typically, you choose a window size $T$ wich is the same size as the anomalies you want to discover, you train on $t-T/2$ and predict $T$ into the future to $t+T/2$. you can then compare your prediction to the real data in a smart way. One way is to use the integral of the error to calculate a normality score. If the error becomes much bigger, the normality score will dip. Very easy in 1D, becomes more difficult in higher dimensions because you need to think of an appropriate mappign to quantify. You have to worry about finding the error both in space and in time.





