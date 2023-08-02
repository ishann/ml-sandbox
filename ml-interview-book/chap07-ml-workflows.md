**NOTE**: GitHub CoPilot helped typeset a lot of the text and most of the equations.

# Chapter 7: Machine Learning Workflows

### 7.1 Basics


> [5] If we have a wide NN and a deep NN with the same number of parameters, which one is more expressive and why?

These questions are difficult to answer without specifics, since the answer usually depends on the task and the network architecture. If we intend to learn compositions of functions, then a deep network may be more expressive. However, "For models initialized with a random, static sparsity pattern in the weight tensors, network width is the determining factor for good performance, while the number of weights is secondary, as long as the model achieves high training accuarcy." - [Golubeva et al., ICLR '21](https://arxiv.org/abs/2010.14495). On the other hand, "We analyze the output predictions of different model architectures, finding that even when the overall accuracy is similar, wide and deep models exhibit distinctive error patterns and variations across classes." - [Nguyen et al.](https://arxiv.org/abs/2010.15327).    
If I had to pick one for being more expressive, I would pick the deep network (deep $\implies$ compositions of functions). However, I would not be surprised if a wide network with the same number of parameters could learn the same function.

> [6] The Universal Approximation Theorem states that a neural network with 1 hidden layer can approximate any continuous function for inputs within a specific range. Then why can’t a simple neural network reach an arbitrarily small positive error?

The Universal Approximation Theorem ([Hornik et al.](https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/notes/Sonia_Hornik.pdf)) "*rigorously establishes that standard multilayer feedforward networks with as few as one hidden layer ... are capable of approximating any ... function ... to any desired degree of accuracy, provided sufficiently many hidden units are available.*"

A few observations:
1. The Theorem requires sufficiently many hidden units, where sufficient may often mean infeasible.
2. The "Approximation" Theorem states that the neural-net can approximate functions to any desired accuracy, but does not provide any error bounds as a function of hidden units. Thus, the desired accuracy may require an arbitrarily large number of hidden units.
3. Achievening a small error would require that the training data is representative of the unseen test distribution. The more pertinent question then becomes: what happens when an infinitely complex distribution meets an infinite number of hidden units?
4. Numerical approximations in real-world neural-nets are likely to lend themselves to small errors.

> [8] Hyperparameters
> [iii] Explain algorithm for tuning hyperparameters.

*Trivial algorithms* involve either a grid search or a random search of the hyperparameter space. Grid search can be prohibitively slow; random search is a reasonable trade-off for grid search.    
*Non-trivial algorithms* involve bayesian optimization or genetic algorithms.    
Even non-trivial algorithms may not give the best results. *In practice*, I have observed that domain experts, after running many dozen or a few hundred experiments, are often able to tune vanilla SGD with hand-set hyperparameters to outperform auto-hyperparameter optimization algorithms.

> [10] Parametric vs. non-parametric methods.
> [ii] When should we use one (parametric methods) and when should we use the other (non-parametric methods)?

(Assuming everyone knows what parametric and non-parametric learning methoda are.)    
Parametric methods are useful for their efficiency with small sample sizes (due to prior modeling assumptions) and ease of interpretation of parameteric representations. Non-parametric methods are useful in handling complex data distributions without assumptions and due to their robustness to outliers.    
Thus, if we have a good understanding of the underlying data distribution and are able to make reasonable assumptions about the distribution, parametric methods may be appropriate. However, non-parametric methods are more appropriate if you have limited knowledge about the underlying data distribution and/or it is a complex distribution with sufficient data to model it.


> [11] Why does ensembling independently trained models generally improve performance?

Complementary independently trained models with diverse learning strategies often compensate for individual models' weaknesses, leading to more accurate and robust predictions. Ensembles help reduce overfitting (overcoming overfitting of individual models), reduce the overall bias (by canceling out individual bias), and lend themselves to learning a more complex additive representation (combining individual predictors).

> [12] Why does L1 regularization tend to lead to sparsity while L2 regularization pushes weights closer to 0?

L1 regularization adds a penalty term to the loss proportional to the sum of the absolute values of the model's coefficients. The isosurface for the L1 norm penalty has sharp corners at the axes, which means that the loss function becomes non-differentiable at zero. This property encourages the model to drive some coefficients to exactly zero, effectively removing corresponding features from the model.

L2 regularization adds a penalty term to the loss function based on the squared values of the model's coefficients. The smoothness of the isosurface for the L2 norm penalty means that there is no "sharp push" towards exactly zero weights, as is the case with L1 regularization. Instead, it gently penalizes large weights, encouraging them to be small but not exactly zero.

> [14] What problems might we run into when deploying large machine learning models?

"Large" ML models may run into the following challenges:    
1. Compute Constraints: May require high-performance GPUs and specialized hardware to support high data and processing throughput.    
2. Inference Latency: High inference times, making real-time applications and low-latency requirements challenging to meet.    
3. Edge Deployment: Large models can be problematic for deployment on devices with limited storage capacity.    
4. Overfitting: Large ML models may overfit to the training distribution, limiting their ability to generalize to real-world distributions.    

> [15] Your model performs really well on the test set but poorly in production.    
> [i] What are your hypotheses about the causes?
 
1. Distribution Shift/ Data Drift: Production data may have a different data distribution than the test set.    
2. Overfitting: Cross-validation sanity may have been violated and the model may have been overfit to the test set.    
3. Data Leakage: During cross-validation, data leakage between train and test sets may result in (inaccurate) high test set performance.    

> [ii] How do you validate whether your hypotheses are correct?    

* Analyze test and production data distributions: Both qualitative (user-based) metrics and quantitative (statistical) metrics may be used to identify data distribution differences.
* Analyze and monitor ML model performance: Apart from tracking model performance, consider tracking proxies such as the confidence in making predictions.

> [iii] Imagine your hypotheses about the causes are correct. What would you do to address them?

* (Model) Active Learning: Query an oracle (whoever labeled test data) for either self-assessed low confidence predictions or known (audited or user feedback-based) incorrect predictions.
* (Model) Lifelong/ Online Learning: Update the model as it sees new (production) data. Can be trickier to implement than Active Learning.
* (Data) Investigate reasons for the data distribution shift and attempt to bridge the gap between the distributions.

### 7.2 Sampling and creating training data

> [2] What is the difference between sampling with vs. without replacement? Name an example of when you would use one rather than the other?

> [3] Explain Markov chain Monte Carlo sampling.


________________________________________________________________


> [4] If you need to sample from high-dimensional data, which sampling method would you choose?

> [5] Suppose we have a classification task with many classes. An example is when you have to predict the next word in a sentence -- the next word can be one of many, many possible words. If we have to calculate the probabilities for all classes, it’ll be prohibitively expensive. Instead, we can calculate the probabilities for a small set of candidate classes. This method is called candidate sampling. Name and explain some of the candidate sampling algorithms ([hint](https://www.tensorflow.org/extras/candidate_sampling.pdf)).