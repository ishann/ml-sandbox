**NOTE**: GitHub CoPilot helped typeset a lot of the text and most of the equations.

# Chapter 7: Machine Learning Workflows

### 7.1 Basics

> [5] If we have a wide NN and a deep NN with the same number of parameters, which one is more expressive and why?

These questions are difficult to answer in general, since it depends on the task and the network architecture. If we intend to learn compositions of functions, then a deep network may be more expressive. However, "For models initialized with a random, static sparsity pattern in the weight tensors, network width is the determining factor for good performance, while the number of weights is secondary, as long as the model achieves high training accuarcy." - [Golubeva et al., ICLR '21](https://arxiv.org/abs/2010.14495). On the other hand, "We analyze the output predictions of different model architectures, finding that even when the overall accuracy is similar, wide and deep models exhibit distinctive error patterns and variations across classes." - [Nguyen et al.](https://arxiv.org/abs/2010.15327).    
If I had to pick one for being more expressive, I would pick the deep network (deep $\implies$ compositions of functions). However, I would not be surprised if a wide network with the same number of parameters could learn the same function.

> [6] The Universal Approximation Theorem states that a neural network with 1 hidden layer can approximate any continuous function for inputs within a specific range. Then why can’t a simple neural network reach an arbitrarily small positive error?

A few observations:
1. The Theorem requires an infinite number of hidden units to approximate continuous functions.
2. The "Approximation" Theorem only states that the neural-net can approximate continuous functions. It does not say anything about error bounds.
3. Achievening a small error would require that the training data is representative of the unseen test distribution. The more pertinent question then becomes: what happens when an infinitely complex distribution meets an infinite number of hidden units?
4. Numerical approximations in real-world neural-nets are likely to lend themselves to small errors.

> [8] Hyperparameters
> [iii] Explain algorithm for tuning hyperparameters.


> [10] Parametric vs. non-parametric methods.
> [ii] When should we use one (parametric methods) and when should we use the other (non-parametric methods)?



