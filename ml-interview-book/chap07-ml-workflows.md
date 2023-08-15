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

Sampling with replacement *replaces* the item into the population (each item can be sampled more than once), while sampling without replacement *removes* the item from the population (each item can be sampled exactly once). Use sampling *with replacement* when the population is significantly larger than the number of sampled items; for example, simulations or bootstrapping. Use sampling *without replacement* when avoiding duplicates to obtain a representative sample of the population; for example, human surveys or experiments

> [3] Explain Markov chain Monte Carlo sampling.

Given a probability distribution ($p(x)$), generally, sampling techniques sample uncorrelated samples from $p(x)$. However, MCMC sampling samples from $p(x)$ by constructing a Markov chain whose stationary distribution is tends towards $p(x)$. The Markov chain is constructed by defining a transition function ($T(x|y)$) and the Markov chain is executed until the *detailed balance condition* ($p(x)T(y|x)=p(y)T(x|y)\hspace{1mm}\forall\hspace{1mm}x, y$) is met.    
MCMC is a class of sampling techniques; popular MCMC methods include the Metropolis-Hastings algorithm, Gibbs sampling, and Hamiltonian Monte Carlo sampling.

> [4] If you need to sample from high-dimensional data, which sampling method would you choose?

Traditional sampling methods like direct sampling or grid-based methods may become infeasible with high-dimensional data due to the curse of dimensionality. Thus, MCMC methods are often used to sample from high-dimensional data distributions. The following are a few of the popular MCMC methods:

1. *Metropolis-Hastings Algorithm*: Samples from new states based on a proposal distribution and builds a Markov chain based on an acceptance criterion.
2. *Gibbs Sampling*: Assumes the distribution to be factorize-able into conditional distributions. Samples one variable at a time while keeping other variables fixed.
3. *Hamiltonian Monte Carlo*: Samples by simulating the movement of a particle by introducing auxiliary Hamiltonian dynamics (momentum) variables, allowing it to explore the parameter space more efficiently.

> [5] Suppose we have a classification task with many classes. An example is when you have to predict the next word in a sentence -- the next word can be one of many, many possible words. If we have to calculate the probabilities for all classes, it’ll be prohibitively expensive. Instead, we can calculate the probabilities for a small set of candidate classes. This method is called candidate sampling. Name and explain some of the candidate sampling algorithms ([hint](https://www.tensorflow.org/extras/candidate_sampling.pdf)).

1. *Sampled Softmax*: Class probabilities are calculated only for a small random subset of the total classes. Candidate class selection happens through uniform random sampling or importance sampling.
2. *Hierarchical Softmax*: Exploits class semantics that lend themselves to a hierarchical structure. Softmax is approximated by traversing the classes organized into a tree to evaluate only a subset of candidate classes. Reduces computation complexity from linear to logarithmic.
3. *Negative Sampling*: Popularized by Word2Vec. Randomly select negative example classes for each positive example. Efficiently learns by training the model to differentiate between true target words and randomly sampled non-target words.
4. *Adaptive Sampling*: Dynamically adjust distribution of candidate classes to focus on hard examples. Sampling distribution may be adjusted to give higher weightage to misclassified classes.

> [6] Suppose you want to build a model to classify whether a Reddit comment violates the website’s rule. You have 10 million unlabeled comments from 10K users over the last 24 months and you want to label 100K of them.    

$100k$ labels over $10k$ users $\implies$ 10 labels per user.    
$100k$ labels over $10mil$ samples $\implies$ 1 label per 100 samples.    
Timeperiod: $24$ months.    

> [i] How would you sample 100K comments to label?    

1. *Active Labeling*: Use active learning to progressively label the most informative/ low confidence predictions, and iteratively learn improved models.
2. *Expert Human-in-the-Loop Labeling*: Involve human expert annotators to manually review comments and prioritize ones more likely to violate the site's rules. Can be used to guide Active Labeling process as well.    
3. *Stratified Sampling + Labeling*:
    1. *By user*: Randomly sample a proportionate number of comments from each user.
    2. *By timeperiod*: Ensure that all temporal trends are captured and labeled for the model to learn.

> [ii] Suppose you get back 100K labeled comments from 20 annotators and you want to look at some labels to estimate the quality of the labels. How many labels would you look at? How would you sample them? [Hint](https://www.cloudresearch.com/resources/guides/sampling/pros-cons-of-different-sampling-methods)

*Determining sample size*: Depends upon desired confidence level, margin of error, and number of annotators.

Sampling Goals:
1. *Random Sampling*: Randomly select comments + labels for evaluation to ensure that the sampled comments are representative of the entire set.
2. *Continuous Monitoring*: Do not wait for 100k samples to be labeled. Continuously evaluate as data is labeled to get a sense of labeling quality/ task difficulty/ etc.
3. *Stratified Sampling by Annotator*: Ensure that each annotator's comments are proportionately sampled.
4. *Establish Inter-Annotator Agreement Metrics*: For each comment, track inter-annotator agreement. Remodel the sampling strategy or labeling process if agreement is low.
5. *Consider Judgement Sampling*: Depending on annotator expertise/ comment representations/ temporal trends, certain samples may be more likely to be mislabeled.

> [7] Suppose you work for a news site that historically has translated only 1% of all its articles. Your coworker argues that we should translate more articles into Chinese because translations help with the readership. On average, your translated articles have twice as many views as your non-translated articles. What might be wrong with this argument?

Only $1\%$ of all articles have been translated $\implies$ some heuristics (topic/ target viewership/ etc.) would have been used by human decision makers to shortlist articles that should be translated. Thus, articles concerned with more popular content may have been the ones that were deemed worth translating $\implies$ there may have been *selection bias* in choosing popular articles to be translated. The fact that they were translated is not the cause of their popularity.    
Instead, randomly select articles for translation and perform AB testing to validate the hypothesis.

> [8] How to determine whether two sets of samples (e.g. train and test splits) come from the same distribution?

To determine whether two sets of samples come from the same distribution, you can use statistical tests and visualizations. Here are some common methods you can use:

1. *(Qualitative) Visual Inspection*: Plot histograms or mass/ density distributions of both sets and check if they are visually similar.
2. *(Quantitative) Statistical Tests*: Either use a Chi-Square Test for categorical data or use a Kolmogorov-Smirnov Test for continuous data to compare the distributions and apply null-hypothesis significance testing.
3. *Machine Learning*: It may be possible to qualitatively measure similarity by treating one as training data and the other as testing data (also swap for a duplicate experiment). Use generalization performance as a measure of similarity between the distributions. However, underfitting/ overfitting due to mismatched model complexity may make this an unreliable method.

> [9] How do you know you’ve collected enough samples to train your ML model?

"Enough" samples depends upon problem complexity, data quality, and the desired level of model performance. A few criterions that affect required sample size:
1. *Problem Complexity*: Complex problems may require more data to learn good representations; for example, Vision and Language tasks may require more data than tabular tasks.
2. *Model Complexity*: More complex models require more data for generalization to avoid overfitting.
3. *Bias and Variance*: If the model has high bias, more data will not help. If the model has high variance, more data could help.

Methods to estimate when we have enough samples:
1. *Cross Validation*: $k$-fold cross-validation over progressively more data samples can indicate when model performance due to more data plateaus.
2. *Statistical Power*: Statistical power analysis can be used to determine the minimum sample size required to detect a given effect size with a given degree of confidence.
3. *VC Dimension*: This may not be practical for complex models, but the VC dimension is a measure of the complexity of the model, and can be used to determine the minimum number of samples required to learn a model with a given degree of confidence.

> [10] How to determine outliers in your data samples? What to do with them?

How to determine outliers:
1. *Visualization*:
   1. *Plots*: Visually inspect histograms, scatter plots, and density plots.
   2. *Box Plots and Inter-Quartile Range*: Observe the median and median, and inter-quartile ranges in box-and-whisker plots to detect outliers.
2. *Z-Scores*: Compute z-score for each sample. Depending on the data distribution and other factors such as noise, a z-score greater than 2 or 3 may indicate an outlier.
3. *DBSCAN*: Density-based clustering algorithms can be used to detect outliers.

What to do with outliers:
1. *Remove*: Remove if we are sure that they do not provide useful information.
2. *Retain*: Retain if the outlier may provide valuable insights or signal unexpected events. Consider applying transformations (logarithmic, square root, Box-Cox, etc.) while retaining useful outliers.
4. *Impute*: Impute the outlier value with a reasonable estimate based on the data distribution.

_______________________________________________________________


> [11] Sample duplication    

> [i] When should you remove duplicate training samples? When shouldn’t you?    

*Removing duplicates can improve generalization* Duplicate samples can bias your model's training and evaluation, leading to overfitting on the subset of duplicate samples. Removing duplicates can help ensure that your model generalizes better to unseen data. The assumption here is that the training data duplicates do not reflect idiosyncracies that also exist in the test data.    
*Retaining duplicates can fix imbalanced datasets* If the training data is imbalanced, removing duplicates may further exacerbate the class imbalance issue, with significantly reduced performance on the minority class(es). If the same duplication patterns exist in the test data as well, then we may want to keep the train data duplicates since they reflect idiosyncracies that also exist in the test data.    
Strong cross-validation is critical while making changes to the benchmark data.

> [ii] What happens if we accidentally duplicate every data point in your train set or in your test set?

The effective number of unique datapoints remains the same.    
*Train set*: Given the same number of training iterations, the number of training epochs halves without affecting the final model performance.    
*Test set*: All standard evaluation metrics will measure the same performance on the duplicated test set.

> [12] Missing data    

> [i] In your dataset, two out of 20 variables have more than 30% missing values. What would you do?    

1. *Data Imputation*: Impute missing values using methods such as mean/ median imputation, interpolation, regression, or using k-nearest neighbors.
2. *Feature Engineering*: Design features that capture the patterns of the missing data; for example, binary indicator variables may directly signal to the model that data is missing.    

Following strict cross-validation driven methodology when dealing with missing data is critical.

> [ii] How might techniques that handle missing data make selection bias worse? How do you handle this bias?

1. *Data Imputation*: Imputing with mean/ median might distort the distribution of variables, leading to bias in learnt parameters and predictions. Strong cross-validation methods or more advanced imputation methods such as regression imputation may help mitigate biases.
2. *Feature Engineering*: Introducing binary indicator variables can affect the model by introducing bias if the missingness phenomenon is correlated with the target variable. Again, strong cross-validation practices and sensitivity analysis can help us measure and mitigate such biases.


________________________________________________


> [13] Why is randomization important when designing experiments (experimental design)?

> [14] [iii] Imagine you want to build a model to detect skin legions from images. In your training dataset, only $1%$ of your images shows signs of legions. After training, your model seems to make a lot more false negatives than false positives. What are some of the techniques you'd use to improve your model?


________________________________________________


> [15] Training data leakage.    
> [i] Imagine you're working with a binary task where the positive class accounts for only 1% of your data. You decide to oversample the rare class then split your data into train and test splits. Your model performs well on the test split but poorly in production. What might have happened?    
> [ii] You want to build a model to classify whether a comment is spam or not spam. You have a dataset of a million comments over the period of 7 days. You decide to randomly split all your data into the train and test splits. Your co-worker points out that this can lead to data leakage. How?    
> Hint: You might want to clarify what oversampling here means. Oversampling can be as simple as dupplicating samples from the rare class.


________________________________________________


> [16] How does data sparsity affect your models?    
> Hint: Sparse data is different from missing data.

> [17] [iii] Feature leakage: How do you detect feature leakage?


________________________________________________


> [18] Suppose you want to build a model to classify whether a tweet spreads misinformation. You have 100K labeled tweets over the last 24 months. You decide to randomly shuffle on your data and pick 80% to be the train split, 10% to be the valid split, and 10% to be the test split. What might be the problem with this way of partitioning?

> [19] You’re building a neural network and you want to use both numerical and textual features. How would you process those different features?

> [20] Your model has been performing fairly well using just a subset of features available in your data. Your boss decided that you should use all the features available instead. What might happen to the training error? What might happen to the test error?    
> Hint: Think about the curse of dimensionality: as we use more dimensions to describe our data, the more sparse space becomes, and the further are data points from each other.


________________________________________________


### 7.3 Objective functions, metrics, and evaluation

> [3] Bias-variance trade-off    
> [ii] How’s this tradeoff related to overfitting and underfitting?    
> [iii] How do you know that your model is high variance, low bias? What would you do in this case?    
> [iv] How do you know that your model is low variance, high bias? What would you do in this case?    


________________________________________________


> [4] Cross-validation.    
> [ii] Why don’t we see more cross-validation in deep learning?

> [5] Train, valid, test splits.    
> [iii] Your model’s loss curves on the train, valid, and test sets look like the image below. What might have been the cause of this? What would you do?    
<img src="https://huyenchip.com/ml-interviews-book/contents/images/image25.png" alt="train-valid-test-splits" width="400"/>


________________________________________________


> [7] F1 score.    
> [ii] Can we still use F1 for a problem with more than two classes. How?    


> [8] Given a binary classifier that outputs the following confusion matrix.    
> |             | Predicted True | Predicted False |
> | ----------- | -------------- | --------------- |
> | Actual True | 30 | 20 |
> | Actual False|  5 | 40 |    
> [ii] What can we do to improve the model’s performance?


________________________________________________


> [9] Consider a classification where 99% of data belongs to class A and 1% of data belongs to class B.    
> [i] If your model predicts A 100% of the time, what would the F1 score be? Hint: The F1 score when A is mapped to 0 and B to 1 is different from the F1 score when A is mapped to 1 and B to 0.     
> [ii] If we have a model that predicts A and B at a random (uniformly), what would the expected F1 be?


________________________________________________


> [10] For logistic regression, why is log loss recommended over MSE (mean squared error)?

> [11] When should we use RMSE (Root Mean Squared Error) over MAE (Mean Absolute Error) and vice versa?


________________________________________________


> [12] Show that the negative log-likelihood and cross-entropy are the same for binary classification tasks.

> [13] For classification tasks with more than two labels (e.g. MNIST with 10 labels), why is cross-entropy a better loss function than MSE?


________________________________________________


> [16] MPE (Most Probable Explanation) vs. MAP (Maximum A Posteriori)    
> [i] How do MPE and MAP differ?    
> [ii] Give an example of when they would produce different results.

> [17] Suppose you want to build a model to predict the price of a stock in the next 8 hours and that the predicted price should never be off more than 10% from the actual price. Which metric would you use? Hint: check out MAPE.

