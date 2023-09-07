# Machine Learning Basics

These notes are taken in no particular order. There is no beginning or end.    
Also, basic notation is not defined. The reader is expected to be familiar with standard ML notation.

## Gradient Descent

Update model parameters ($\theta_k$) after oberving $(x_i, y_i, \hat{y_i})$ tuples.    
Magnitude + direction of the update is in the opposite direction of the gradient: $\Delta \theta = -\eta \nabla L(\theta)$.    
Thus, $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$.

The loss, $L(\theta_t)$, is a function of the $(x_i, y_i, \hat{y_i})$ tuples and is discussed below.    

Traditionally, gradient descent updates are expected to be performed over the entire training distribution. However, a few practical variations are discussion below.

### Batch Gradient Descent

Repeat _until convergence_:    
    $\hspace{10mm}\theta_j := \theta_j + \Sigma_{i=1}^m L(x_i, y_i, \hat{y_i}) \hspace{4mm}\forall j$

### Stochastic Gradient Descent (SGD)

Loop over training examples and repeat _until convergence_:    
    $\hspace{10mm}$ for $i=1$ to $m$:    
    $\hspace{20mm}\theta_j := \theta_j + L(x_i, y_i, \hat{y_i}) \hspace{4mm}\forall j$

### Mini-batch SGD
Trade-off between batch GD and stochastic GD. Converges faster than batch GD due to more frequent parameter updates, while vectorized ops provide computational performance gains over SGD.

Loop over training examples in batches and repeat _until convergence_:    
    $\hspace{10mm}$ for $i=1$ to $m$ in batches of size $k$:    
    $\hspace{20mm}\theta_j := \theta_j + \Sigma_{i=1}^k L(x_i, y_i, \hat{y_i}) \hspace{4mm}\forall j$

**NOTE**: A probabilistic interpretation of gradient descent lends itself to allow us to model $y_i = \theta^T x_i + \epsilon_i$, where $\epsilon_i$ may be drawn from a suitable probability distribution (usually Gaussian distribution). $\epsilon_i = y_i-\theta^T x_i \implies p(\epsilon_i)=p(y_i| x_i; \theta_i)$. Thus, $L(\theta)=L(\theta, \bar{X}, \bar{y})=p(\bar{y}|\bar{X}; \theta)$ lends itself to a maximum likelihood estimation problem.

## Loss Functions

* **Binary Cross-Entropy Loss (Log Loss)**:    
  Measures dissimilarity between predicted class and true label.    
  BCE is defined as the negative log likelihood for the binary classification problem:    
  $L_{\text{BCE}} = -y_i \log(p_i) - (1-y_i) \log(1-p_i)$.

* **Categorical Cross-Entropy Loss**:    
  An extension of cross-entropy loss for multi-class classification problems.    
  Generally used with softmax activation function in the output layer of a neural network for numerical stability.    
  $L_{\text{CE}}$ =  $-\Sigma_i y_i \log(p_i)$.    
  

* **Hinge Loss**:    
  Used for optimizing maximum-margin classifiers such as SVMs.    
  $L_{\text{Hinge}} = \sum_{i=1}^{k} \max\left(0, 1 - y_i \cdot \hat{y_i}\right)$    
  (`NOTE`: $\hat{y_i}$ is the raw system prediction).

* **Mean Squared Error (MSE)**:    
   Used in regression problems and as a model parameter regularizer. Is suspectible to outliers.    
   $L_{\text{MSE}} = \frac{1}{n} \Sigma_{i} (y_i - \hat{y_i})^2$

* **Mean Absolute Error (MAE)**:    
  Used in regression problems and as a model parameter regularizer. Robust to outliers. Results in sparse solutions.    
  $L_{\text{MAE}} = \frac{1}{n} \Sigma_{i} |y_i - \hat{y_i}|$

* **Huber Loss**:    
  Combination of $L_{\text{MSE}}$ and $L_{\text{MAE}}$.    
  $L_{\delta} = \frac{1}{2}{(y_i - \hat{y_i})^2}$ if $|y_i - \hat{y_i}|\leq\delta$ else $\delta\cdot\left(|y_i-\hat{y_i}|-\frac{1}{2}\delta\right)$    
  Smoothly transitions between MSE and MAE based $\delta$.

* **Kullback-Leibler Divergence (KL Divergence)**:    
  Measures the difference between two probability distributions.
  Used for optimizing VAEs and GANs, and also for tSNE.    
  $D_{KL}= \Sigma(y_i \cdot \log(y_i / \hat{y_i}))$


References:
1. See PyTorch's [loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions) for reference implementations.
2. Also, see [Metrics@ML-System-Design](./ML-System-Design.md#metrics).


## Optimization

SGD, then adaptive ones.

## Supervised Learning

### Logistic Regression

Borrow from linear regression machinery. Modify it so that the output is a probability distribution over the classes, i.e., $0\leq\hat{y_i}\leq1$.

$h_\theta(x)=g(\theta^Tx)=\frac{1}{1+{\rm e}^{-\theta^{T}x}}$, where $g(z)=\frac{1}{1+{\rm e}^{-z}}$ is the logistic or sigmoid function.

Interesting aside (which makes taking derivatives easy): $g'(z)=g(z)(1-g(z))$.

Taking the negative log likelihood of the above expression results in the BCE loss expression.

### Naive Bayes
$\text{Posterior} = \frac{\text{Likelihood}\cdot\text{Prior}}{\text{Evidence}} \implies P(Y|X) = \frac{P(X|Y)\cdot P(Y)}{P(X)} = \frac{P(X|Y)\cdot P(Y)}{\Sigma_{y\in Y}P(X|Y)}$

MAP (Maxium A Posteriori) Rule: $\hspace{2mm}\hat{y} = \underset{k\in{1,2,...,K}}{\operatorname{argmax}} \hspace{2mm}p(y_k) \cdot \Pi_{i=1}^{n} p(x_i|y_k)$    
`NOTE`: The MAP Rule is independent of the evidence term, $P(X)$.

### Decision Trees (CART)

**Basic**
* Consider all features. Examine different split points using a cost function. Greedily select split with the optimal cost. Recurse.

* Splitting criterions:    
  Too many splits will cause overfitting.
  - Classification
    - Gini Impurity: Measures degree of impurity in a dataset. Lower is better.    
    $\text{Gini}(D) = 1 - \sum_{i=1}^{C} (p_i)^2$,    
    - Entropy: Measures uncertainty in a dataset. Lower is better.    
    $\text{Entropy}(D) = - \sum_{i=1}^{C} p_i \log_2(p_i)$,    
   where $D$ is the training data, $C$ is the number of classes, and $p_i$ is the proportion of samples in class $i$ in $D$.    
  - Regression
    - Mean Squared Error: Lower is better.    
    $\text{MSE}(D) = \frac{1}{|D|} \sum_{i \in D} (y_i - \bar{y})^2$,    
    - Mean Absolute Error: Lower is better.    
    $\text{MAE}(D) = \frac{1}{|D|} \sum_{i \in D} |y_i - \bar{y}|$,    
    where $D$ is the dataset being split, $y_i$ is the target value for sample $i \in D$, and $\bar{y}$ is the mean target value for all samples in $D$.

* Stopping criterions:
  - Max Depth: Upper bound the longest path length from root to leaf.
  - Min Samples Split: Lower bound the minimum training inputs required at a new split.
  - Pruning: Remove low importance sub-trees based on pre-defined criterions.

**Extensions**

* ID3
  - Information Gain
    - Criterion to quantify split quality.
    - Measures the reduction in entropy after a split. Higher is better.
  - $\text{IG}(S,A)=\text{Entropy}(S)–\Sigma(\frac{|S_v|}{|S|})\times \text{Entropy}(S_v)$,    
    where, $S$ is the dataset, $A$ is a feature, $S_v$ is the subset of $S$ for which $A$ takes the value $v$.
  - Algorithm:    
    - Calculate the entropy of every attribute $a$ of $S$.
    - Split $S$ into subsets using the attribute for which the resulting $\text{IG}$ is maximum, resulting in a new node.
    - Recurse on subsets using the remaining attributes (considering only attributes never selected before.).

* C4.5
  -  Splitting criterion is normalized $\text{IG}$:    
     $\text{Gain Ratio}(A) = \frac{\text{IG}(A)}{\text{SI}(A)}$,
     where $\text{SI}(A)$, the Split Information, is a measure of the potential information generated by splitting on attribute $A$.    
     $\text{SI}(A) = -\sum_{v \in \text{values}(A)} \frac{|S_v|}{|S|} \cdot log_2\left(\frac{|S_v|}{|S|}\right)$
  -  $\text{SI}(A)$ accounts for $\#$ attribute values. Mitigates bias towards attributes with many values, leading to balanced trees.
  -  Handles continuous features.
  -  Runs through the tree once it has been created for pruning. Also, handles missing values via additional branching.

### Random Forests

Bootstrap aggregagtion (bagging) of ensembled decision trees.    

* Bootstrap Aggregation (Bagging): Create multiple dataset by sampling with replacement, resulting in diverse decision trees; thus, reducing overfitting.    
* Feature Randomization: Randomly sample a subset of features for building trees, leading to diverse trees and preventing few features from dominating.    
* Voting or Averaging: For classification, majority voting is effective. For regression, average of individual predictions is used.    

### Boosting

* Bagging reduces variance (overfitting). Boosting reduces both bias and variance.
* Weighted voting (vs. constant voting in bagging) results in better models having more weightage.
* Emphasizes misclassified data points in each iteration.
* AdaBoost (Adaptive Boosting)
  * In each training round, give more weight to misclassified training instances.
  * Ensemble of weak learners is weighted based on training misclassification error.
* Gradient Boosting
  * In each training round, a new base learner is fitted to the residuals/ difference between targets and predictions.
  * A learning rate (often termed $\eta$) controls contributions of each new base learner by scaling the contribution of each base learner, preventing overfitting, and allowing smoother convergence.
  * The weighted sum of the base learners' predictions, where weights are determined by the learning rate, represents the final prediction.
The new base learner is trained to minimize the resid

### K-Nearest Neighbors
* Non-parameteric supervised learning algorithm.
* Predict by looking at the $k$ nearest neighbors from the training data (voting for classification, mean for regression).
* Small $k$ leads to noise-sensitive but flexible models. Large $k$ leads to robust but oversmooth decision boundaries.
* Distance-weighted predictions: closer neighbors have higher influence.

### K-Means Clustering
* Unsupervised learning algorithm. Goal is to find cluster centroids that minimize within-cluster variation over all data.
* Algorithm
  * Randomly initialize $k$ cluster centers.
  * Assign each data point to the cluster whose centroid is closest to it.
  * Update cluster centroids by computing the mean of all corresponding cluster data points.
  * Repeat until convergence.
* Convergence criterions
  * No change in cluster assignments/ cluster centroids.
  * Loss function below a threshold. 
  * Max iterations reached.
* Loss function: The "inertia" or "within-cluster sum of squares" (WCSS).    
  $WCSS = \sum_{i=1}^{k} \sum_{j=1}^{n_i} \left\| x_{ij} - c_i \right\|^2$,    
  where $k$ is $\#$ clusters, $n_i$ is $\#$ points in cluster $i$, $x_{ij}$ represents $j^{th}$ point in $i^{th}$ cluster, and $c_i$ is the $i^{th}$ cluster centroid.

### Support Vector Machines (SVMs)
`TODO`

## Unsupervised Learning

### Principal Component Analysis (PCA)
`TODO`

### Singular Value Decomposition (SVD)
`TODO`

### Independent Component Analysis (ICA)
`TODO`

### Linear Discriminant Analysis (LDA)
`TODO`

## Learning Theory
`TODO`

## Bias-Variance Tradeoff /  Underfitting-Overfitting
`TODO`

## PAC Learning
`TODO`

### Acknowledgements
_Most_ of my understanding of ML basics is built upong Andrew Ng's wonderful CS229 notes.
