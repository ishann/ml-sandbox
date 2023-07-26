**NOTE**: GitHub CoPilot helped typeset a lot of the text and most of the equations.

# Chapter 5: Math

## Resources:
1. [CS229 Linear Algebra Primer](https://cs229.stanford.edu/section/cs229-linalg.pdf).
2. [CS229 Probability Primer](https://cs229.stanford.edu/lectures-spring2022/cs229-probability_review.pdf).

## Algebra and (a little) Calculus

### 5.1.1 Vectors

>[2ii] Outer Product: Give an example of how the outer product can be useful in ML.

The outer product of two vectors, $X\otimes Y=Z$,  will capture the relationships between the individual elements of the two vectors: $Z_{ij}=A_i\times B_j$.    
This can be useful when $[i]$ computing covariance matrices for dimensionality reduction, and $[ii]$ several SVM kernel functions may employ outer products to quantify similarity between two data-points.


>[4] Given two sets of vectors $A=a_1, a_2, ..., a_n$ and $B=b_1, b_2, ..., b_m$, how do you check that they share the same basis?

Try to express each individual $a_i \forall i\in\{1,2,...,n\}$ as a linear combination of the $b_j$s. Repeat this process by swapping $A$ and $B$ (checking one way is not sufficient). If both $A$ and $B$ sets can be expressed as linear combinations of the other set, then they share a basis.

>[5] Given $n$ vectors, each of $d$ dimensions, what is the dimension of their span?

If vectors are linearly independent, span=$\min (n, d)$. If vectors are linearly dependent, span may be less than $n$. One thing that can be said for sure: span $\leq d$.

>[6ii] Norms and metrics: How do a norm and a metric differ? Given a norm, make a metric. Given a metric, can we make a norm?

A norm measures the magnitude or length of a vector. f(vector/point) → non-negative value. Properties: non-negativity, homogeneity, triangle inequality.    
A metric measures the separation or dissimilarity between points. f(pairs of vectors/ points) → non-negative value. Properties: non-negativity, symmetry, triangle inequality.    
Norm → Metric. If f is a norm operating on $x$, replace $x$ with $x-y$ to construct a metric. No, given a metric, we may not always be able to construct a norm (metric properties may be violated).


### 5.1.2 Matrices

Skipped 5.1.2 for now.

### 5.1.3 Dimensionality Reduction
>[3] Name some applications of eigenvalues and eigenvectors.

PCA uses eigenvalues and eigenvectors to map high-dimensional inputs into a lower-dimensional latent space while preserving information (depending on the number of eigenvectors retained).    
Collaborative filtering algorithms use eigen analysis to factorize big (input) data  into latent features that represent user preferences and item characteristics.

>[4] We want to do PCA on a dataset of multiple features in different ranges. For example, one is in the range 0-1 and one is in the range 10 - 1000. Will PCA work on this dataset?

No. The feature with a larger range may dominate the component analysis.    
We pre-process the data by normalizing ($mean=0, std=1$) all features before applying PCA. 

>[5] Under what conditions can one apply eigendecomposition? What about SVD? What is the relationship between SVD and eigendecomposition? What’s the relationship between PCA and SVD?

Eigendecomposition ($A=V\Lambda V^T$) can be applied when the matrix $(i)$ is square, and $(ii)$ is diagonalizable.    
SVD ($A=U\Sigma V^T$) can be applied to all real matrices.    
Given the eigendecomposition of a matrix, we can obtain the SVD of the matrix using $(i)\hspace{1mm}\Sigma=\Lambda$, and $(ii)\hspace{1mm}U=V$.    
PCA can be performed using SVD: $(i)$ normalize data, $(ii)$ compute covariance matrix, $(iii)$ compute SVD of covariance matrix: $C=V\Sigma V^T$ (note that $C$ is a symmetrix matrix resulting in $U=V$), and finally $(iv)$ $V$ has the eigenvectors and $\Sigma$ has the eigenvalues along its diagonal.    
[Relevant discussion about eigen-decomposition and SVD.](link)

>[6] How does t-SNE (T-distributed Stochastic Neighbor Embedding) work? Why do we need it?

How?    
Measure pairwise similarities between data points using a Gaussian kernel.    
Construct a joint probability distribution over pairs of points.    
In low-dimensional space, create a similar joint probability distribution using gradient descent to minimize KL divergence between the distributions.     

Why?    
$(i)$ Captures non-linear relationships.    
$(ii)$ Preserves both local and global structure in the low-dimensional projection.

### 5.1.4 Calculus and Convex Optimization

>[1] Differentiable functions: Give an example of non-differentiable functions that are frequently used in machine learning. How do we do backpropagation if those functions aren’t differentiable?

ReLU and max-pooling are non-differentiable.    
In general: Non-diff functions can allow backprop to happen using subgradients or gradient approximations.

Specifically, for the two examples:    
$(i)$ ReLU: Not differentiable at $x=0$, but differentiable everywhere else. Workaround: ReLU derivative considered either $0$ or $1$ at $x=0$. Practically, no floating value is ever exactly 0.    
$(ii)$ MaxPool: Gradients passed only to max value, others receive 0 gradients ("argmax" trick).    

>[2] Convexity: Show that the cross-entropy loss function is convex.

For K-way multi-classification, the cross-entropy loss is: $L=-\sum_{i}^{K}y_i\log{\hat{y}_i}$.    
However, the summation (for each datapoint) reduces to $L=-\log{x}$ since $y=1$ for the correct class and $y=0$ for all other classes.    
$\frac{\partial^2 L}{\partial x^2}=\frac{1}{x^2}\geq 0$.    
Thus, $L$ is convex for each datapoint.    
Summation of $L$ over all datapoints will also be convex.

>[4] Most ML algorithms we use nowadays use first-order derivatives (gradients) to construct the next training iteration.    
>[ii] What are the pros and cons of second-order optimization.    

Pros: $(i)$ Faster convergence, $(ii)$ implicit adaptive learning rates.    
Cons: $(i)$ Computational and memory requirements for modern datasets, $(ii)$ sensitivity to Hessian approximations for complex non-convex objective functions.    

>[iii] Why don’t we see more second-order optimization in practice?

This question may soon become outdated. Recent optimization research (especially a few papers coming out of Google Research) suggest that modern second-order optimizers may be able to overcome computational and memory constraints and be applicable in real-world settings.

>[5] How can we use the Hessian (second derivative matrix) to test for critical points?

1. Compute first-order partial derivatives. Set derivatives equal to zero and solve to obtain potential critical points.
2. Evaluate Hessian at each potential critical point.
3. Examine eigenvalues of Hessian: Compute eigenvalues of Hessian at each critical point. If all eigenvalues are +ve, critical point corresponds to a min. If all eigenvalues are -ve, critical point corresponds to a max. If both +ve and -ve eigenvalues exist, the critical point corresponds to a saddle point. If some eigenvalues are zero, test is inconclusive.

>[9] Given the function $f(x,y)=4x^2−y$ with the constraint $x^2+y^2=1$, find the function’s maximum and minimum values.

Lagrangian: $L(x,y,\lambda)=(4x^2−y)+\lambda(x^2+y^2−1)$

Partial derivatives:
1. $\frac{\partial L}{x}=8x+2xy^2\lambda$
2. $\frac{\partial L}{y}=-1+2x^2y\lambda$
3. $\frac{\partial L}{\lambda}=x^2+y^2-1$

The above SoE results in 4 solutions, which when plugged into $f(x,y)$ provides min and max values of $-1$ and $\frac{65}{16}$.

<span style="color:DodgerBlue">NOTE: From this point onwards, I decided to not spend time on solving numericals and instead focus on Medium/ Hard conceptual questions.</span>.

## 5.2 Probability and Statistics

### 5.2.1 Probability

>[7] Is it possible to transform non-normal variables into normal variables? How?

It is possible.    
First, figure out whether the variable follows a normal distribution either by visual inspection (histogram/ boxplot/ etc.) or by using the Kolmogorov Smirnov test.    
Then, apply a Power Transform (after ensuring all variable values $>0$). Relevant function: `scipy.stats.boxcox`.

>[8] When is the t-distribution useful?

When sample size is small and standard deviation is not known (and cannot be reliably estimated from small data sample).    
Rule of thumb is to use t-distribution when 30 or fewer data samples are available. Else, use Normal distribution.

> [11] Given two random variables $X$  and $Y$, we have $P(X|Y)$ and $P(Y) \forall X, Y$. How would you calculate $P(X)$?

$P(X)=\sum_{Y_i\in Y} P(X|Y_i)P(Y_i)$

> [16] Consider a person A whose sex we don’t know. We know that for the general human height, there are two distributions: the height of males follows $h_m=\mathcal{N}(\mu_m,\sigma_m)$  and the height of females follows $h_j=\mathcal{N}(\mu_j,\sigma_j)$. Derive a probability density function to describe A’s height.

Assuming that the probability of being a male in the population is $p$, the mixture distribution describing A's height will be: $ph_m+(1-p)h_j$.

> [18] Given $n$ samples from a uniform distribution $[0,d]$, how do you estimate $d$? (Also known as the German tank problem)

[German Tank Problem](https://en.wikipedia.org/wiki/German_tank_problem)    
Error 404: Solution not found.    
This question is beyond the scope of my current preparation.

> [21] You decide to fly to Vegas for a weekend. You pick a table that doesn’t have a bet limit, and for each game, you have the probability p  of winning, which doubles your bet, and 1−p  of losing your bet. Assume that you have unlimited money (e.g. you bought Bitcoin when it was 10 cents), is there a betting strategy that has a guaranteed positive payout, regardless of the value of p ?

(Assuming we have a _lot_ of money, and not _infinite_ money.)    
No. A super-bad streak of losses might make you lose all money before you are able to leverage a high $p$ to leave with a positive payout.    

(Assuming we have _infinite_ money.)    
Apply the ''Martingale strategy'':    
1. Start with an initial bet amount, $I$.
2. If we win, restart with $I$.
3. If we lose, double the amount and continue playing.
4. Repeat Step 3 until we win.

The idea behind the Martingale strategy is that we will eventually win and that one win covers all previous losses.

>[22] Given a fair coin, what’s the number of flips you have to do to get two consecutive heads?


Let $X$ be the expected number of flips for two consecutive heads.    
Let's model the first two tosses through 4 equally likely events ($P(X_i)=0.25$):    
$\hspace{8mm}X_1:TT...$, $X_2:TH...$, $X_3:HT...$, and $X_4:HH...$.    
$X_{1}$ has a $T$ in the first toss and results in us resetting our event of interest by 1.    
$X_{2}$ also has a $T$ in the first toss and results in us resetting our event of interest by 1.    
$X_3$ has a $T$ in the second flip and results in us resetting our event of interest by 2.    
$X_4$ represent two consecutive heads and represents the event of interest.    
Thus, $X = 0.25(X+1)+0.25(X+1)+0.25(X+2)+0.25(2) \implies X=6$.


>[23] In national health research in the US, the results show that the top 3 cities with the lowest rate of kidney failure are cities with populations under 5,000. Doctors originally thought that there must be something special about small town diets, but when they looked at the top 3 cities with the highest rate of kidney failure, they are also very small cities. What might be a probabilistic explanation for this phenomenon?

Small samples cannot be used to estimate population parameters since even small (random) changes may result in large fluctuations in the parameter estimate.    
It is likely that the 6 cities stood out on the extrema of the distribution by chance.

>[24] Derive the maximum likelihood estimator of an exponential distribution.

$f(x,\lambda)=\lambda e^{-\lambda x} \hspace{2mm}\forall\hspace{1mm} x\in X$    
, where $\lambda$ is the parameter whose MLE we are interested in and $X$ is the domain of the random variable $x$.    

The Likelihood function for $f(x,\lambda)$ over $\{x_1, x_2, ..., x_n\}$ is: $\mathcal{L}(\lambda)=\prod_{i=1}^{n} \lambda e^{-\lambda x_i}=\lambda^n e^{-\lambda \sum_{i=1}^{n} x_i}$.    
Log Likelihood: $\ln \mathcal{L}(\lambda)=n\ln \lambda - \lambda \sum_{i=1}^{n} x_i$.    
Setting derivative of Log Likelihood to 0:
$\frac{\partial \ln \mathcal{L}(\lambda)}{\partial \lambda}=\frac{n}{\lambda}-\sum_{i=1}^{n} x_i=0 \implies \lambda=\frac{n}{\sum_{i=1}^{n} x_i}$.    
Thus, MLE estimate for Exponential Distribution: $\lambda=\frac{n}{\sum_{i=1}^{n} x_i}$.


### 5.2.2 Stats
> [3] When should we use median instead of mean? When should we use mean instead of median?

The mean is useful when:
1. Data distribution is symmetric and there are no outliers.
2. Data type is numeric or ordinal.
3. The mean is more efficient to compute compared to the median ($O(n)$ vs. $O(n\log{n})$).

The median is useful when:
1. Outliers result in the mean no longer being a measure of central tendency.
2. Data distribution is skewed and (again) the mean is no longer a measure of central tendency. 
3. If data type is nominal.

> [4] What is a moment of function? Explain the meanings of the zeroth to fourth moments.    

The moment of a function is a quantity that describes the function's shape/ behavior, quantifying how the function's values will be distributed around a point or an axis.    
1. The $0^{th}$ moment is the area under the curve of the function, representing the total mass or probability of the function.
2. The $1^{st}$ moment is the center of mass of the function, representing the expected value or the mean.
3. The $2^{nd}$ moment is the variance, representing the spread or dispersion of the function from the first moment. 
4. The $3^{rd}$ moment is the skewness of the function, representing the asymmetry of the function (quantifying the skew to the left or the right of the mean). 
5. The $4^{th}$ moment is the kurtosis of the function, representing the shape of the function (quantifying the heaviness of the tails and the peakness of the distribution relative to a normal distribution).    
`NOTE`: I have _never_ used the $3^{rd}$ and $4^{th}$ moments in practice.

> [5] Are independence and zero covariance the same? Give a counterexample if not.

Covariance only checks for linear dependence between two variables.
Independence checks for _any_ type of dependence between two variables.    
Example: Let $x\in X$ be drawn from a standardized normal distribution. $f_1=x$ and $f_2=x^2$ have 0 covariance but are clearly not independent. 

> [7] Suppose that we examine 100 newborn puppies and the 95% confidence interval for their average weight is  $[0.9,1.1]$  pounds. Which of the following statements is true?    
> [i] Given a random newborn puppy, its weight has a 95% chance of being between 0.9 and 1.1 pounds.    
> [ii] If we examine another 100 newborn puppies, their mean has a 95% chance of being in that interval.    
> [iii] We're 95% confident that this interval captured the true mean weight.

''A confidence interval refers to the long-term success rate of the method, that is, how often this type of interval will capture the parameter of interest.'' - [Khan Academy](https://www.khanacademy.org/math/ap-statistics/xfb5d8e68:inference-categorical-proportions/introduction-confidence-intervals/a/interpreting-confidence-levels-and-confidence-intervals).    
[i] is referring to samples and the sample mean.    
[ii] is referring to a future sampling.    
[iii] is correct based on the above definition of confidence intervals.

> [8] Suppose we have a random variable $x$ supported on $[0,1]$ from which we can draw samples. How can we come up with an unbiased estimate of the median of $x$?

It is not possible to obtain an unbiased estimate of the median of a random variable by sampling (since we cannot be sure that the distribution is not skewed).    
Counterfactual: For a fixed median, we can transform the probability density on both sides of the median. Any estimator where mean equals median will have a corresponding alternate distribution where mean does not equal median, thus making the estimator biased.    
Reference: [stats.statexchange.com](https://stats.stackexchange.com/a/36177)


> [9] Can correlation be greater than 1? Why or why not? How to interpret a correlation value of 0.3?

No, correlation cannot be greater than $1$. It will be equal to $1$ when $X$ and $Y$ are identical (upto a multiplicative factor).    
Geometric view: Let $X$ and $Y$ be vectors in Euclidean space. The correlation between $X$ and $Y$ is the cosine of the angle between the to vectors.    
Thus, a correlation value of $0.3$ corresponds to an angle $\theta=\cos^{-1}{0.3}$ between $X$ and $Y$.

> [11] Tossing a coin ten times resulted in 7 heads and 3 tails. How would you analyze whether the coin is fair?

Use the binomial test.
Null hypothesis: P(head-fair)$=\frac{1}{2}$    
P(head-observed) = $^{n}C_k \times p^k \times (1-p)^{n-k} = ^{10}C_7 \times 0.5^7 \times (1-0.5)^{10-7} = 0.12$.    
Traditionally, online software would take P(head-true) and P(head-observed) as input and compute the $p-val$.    
Instead, consider using `scipy.stats.binomtest` to compute the significance of the null hypothesis, which yields $p_{val}=0.34$.    
Since $p_{val}$ is fairly high, we accept the null hypothesis, ie, deem the coin to be fair.

`NOTE`: 10 tosses isn't really a lot of tosses. Observing 70 heads in 100 tosses would have resulted in strongly rejecting the null hypothesis ($7.85e^{-05}$).

> [12] Statistical significance.    
> [i] How do you assess the statistical significance of a pattern whether it is a meaningful pattern or just by chance?    
> [ii] What’s the distribution of p-values?    
> [iii] Recently, a lot of scientists started a war against statistical significance. What do we need to keep in mind when using p-value and statistical significance?    

[i] Null-hypothesis statistically significance testing uses the *p-value* to determine if a result is meaningful pattern or just by chance. The p-value represents *the probability of obtaining test results at least as extreme as the result actually observed, under the assumption that the null hypothesis is correct* - [Wikipedia](https://en.wikipedia.org/wiki/P-value).    
[ii] The distribution of p-values is uniform (when the null-hypothesis is true). See [this](https://stats.stackexchange.com/a/10617) for a discussion on the conditions under which this statement holds.    
[iii] **(a)** Effect Size: Measure strength of the relationship between variables; consider the effect size alongside the p-value to understand the practical significance of the findings. Small p-values with tiny effect sizes may be inconsequential. **(b)** Context and Interpretation: Interpret statistical significance in the context of the RQ and the study design, conditioned on prior knowledge, theoretical expectations, and a holistic understanding of the field of study. **(c)** Reproducibility: Reproducibility is non-negotiable. Use complementary methods (to p-val) to assess the robustness of findings (cross validation, bootstrapping, etc.).

> [13] Variable correlation.    
> [i] What happens to a regression model if two of their supposedly independent variables are strongly correlated?    
> [ii] How do we test for independence between two categorical variables?    
> [iii] How do we test for independence between two continuous variables?

[i] Multi-collinearity causes overfitting and makes it difficult to interpret the regressor. Fix by removing/ combining variables, using regularization, or applying dimensionality reduction (which implicitly combines variables).    
[ii] Apply the Chi-squared Test. [Wikipedia](https://en.wikipedia.org/wiki/Chi-squared_test).    
[iii] Quite difficult to solve for the general case. To begin, inspect scatter plots. Computing Pearson's correlation co-efficient can help (0 correlation *may* be independence). Distance correlation, implemented in [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.correlation.html), goes beyond Pearson's correlation co-efficient in measuring independence.

> [15] You want to test which of the two ad placements on your website is better. How many visitors and/or how many times each ad is clicked do we need so that we can be 95% sure that one placement is better?

Run an A/B test.    
The Q asks us to test if ad A results in a higher click-through rate (CTR) than ad B with $p<0.05$.    
Run statistical hypothesis testing using a one-tailed independent t-test, where $H_0:\mu_A=\mu_B$ and $H_{alt}:\mu_A>\mu_B$.

> [16] Your company runs a social network whose revenue comes from showing ads in newsfeed. To double revenue, your coworker suggests that you should just double the number of ads shown. Is that a good idea? How do you find out?

A linear relationship between revenue and number of ads shown is too simplistic an assumption. It depends on ad relevance, users' ad tolerance, users' ad engagement ads, etc.    
Run an A/B test with revenue as the target variable, while also measuring user engagement. Mixed-methods research can be helpful sometimes; without significantly affecting user experience, I would also consider getting direct feedback from a subset of the users on the number of ads shown.

> [18] How are sufficient statistics and Information Bottleneck Principle used in machine learning?

Sufficient Statistics: Reduce dimensionality of data to statistical information that is necessary for estimating a particular parameter or making predictions. Useful for feature extraction, data compression, 
concisely representing probabilistic models, etc.    
Information Bottleneck Principle: Retain critical information while discarding irrelevant details, i.e., a tradeoff between compression (compress data to extract essential information) and prediction (retain enough information to allow for accurate prediction/ reconstruction).    
At a high level, both identify compact and informative statistics/ representations of the input data that are critical for representation learning tasks (prediction/ reconstruction).
