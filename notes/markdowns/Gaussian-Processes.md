# Gaussian Processes

* Gaussian Processes (`GP`s) allow us to make predictions about data by incorporating prior knowledge. For a given set of training points, there are potentially infinitely many functions that fit the data. `GP`s offer an elegant solution by assigning a probability to each of these functions. The mean of this probability distribution then represents the most probable characterization of the data. Furthermore, using a probabilistic approach allows us to incorporate the confidence of the prediction into the regression result.
* A random variable $\mathbf{x}\in\mathbb{R}^{n}$ follows a multivariate gaussian dist in $n$ dimensions with mean $\mu\in\mathbb{R}^{n}$ and covariance $\Sigma\in\mathbb{R}^{n \times n}$ if: $\newline\hspace*{16mm}p(\mathbf{x};\mu,\Sigma) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x}-\mu)^T \Sigma^{-1} (\mathbf{x} - \mu) \right)$
* A `GP` is an extension of the multivariate gaussian to infinite dimensions. Given an input vector $x\in\mathbb{R}^{n}$, the `GP` returns a transformed vector $y\in\mathbb{R}^{n}$. Every component of $y$ represents the _probability_ of observing $x_i$ according to some Gaussian living in dimension $i$.
* Since a GP acts as a probability distribution, we can also sample new $y$ for new $x$ apart from transforming an $x$ into a $y$. Hence, if we sample the entire domain of $x$, we can obtain the function $f$ such that $f(x)=y$.

**Marginalization**: Integrate along one of the dimensions of the Gaussian to get the probability distribution over the remaining dimensions: $\hspace*{24mm}P(X)\hspace{-1mm}=\hspace{-1mm}\int_{-\infty}^{\infty}P(X,Y)dY$ or $P(X)\hspace{-1mm}=\hspace{-1mm}\Sigma_{Y}P(X,Y)$

**Conditioning**: Updates the distribution over function values at new points, given observations at known points. By conditioning on observed data, `GP`s allow us to effectively incorporate priors and make informed predictions in uncertain scenarios:  
$\hspace*{16mm}X|Y\sim\mathcal{N}(\mu_X+\Sigma_{XY}\Sigma_{YY}^{-1}(Y-\mu_Y), \Sigma_{XX}-\Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX})$
$\newline$
$\hspace*{16mm}Y|X\sim\mathcal{N}(\mu_Y+\Sigma_{YX}\Sigma_{XX}^{-1}(X-\mu_X), \Sigma_{YY}-\Sigma_{YX}\Sigma_{XX}^{-1}\Sigma_{XY})$  
^^ Complex math, but note that the updated means only depend on the conditioned variable, while the covariance matrix is independent of the variable.


#### Functional view of `GP`s

  * `GP`s (which are stochastic processes) are a set of random variables. Let each of these random variables have a corresponding index $i$, which represents the $i^{th}$ dimension of the $n$-dimensional multivariate distribution.
  * **Goal**: Learn the underlying distribution from _train data_ ($Y$).
  * `GP`s attempt to model the underlying distribution of the _test data_ ($X$), by modeling the joint distribution $P_{X,Y}$ - the span of possible function values - as a multivariate normal distribution whose dimensionality is $|X|\hspace{-1mm}+\hspace{-1mm}|Y|$).
  * Regressing on $X$ requires modeling this as Bayesian inference, i.e., update the (current) hypothesis as information ($Y$) becomes available. This conditional probability ($P_{X|Y}$) is a multivariate normal distribution as well.
  * `GP`s treat each test point as a random variable. Making predictions requires sampling from the distribution; the $i^{th}$ component of the sampled vector is interpreted as the function value corresponding to the $i^{th}$ test point.

Setting up the distribution requires defining $\mu$ and $\Sigma$. In `GP`s, centering the data, i.e., $\mu=0$ is common practice. The covariance matrix ($\Sigma$) is determined by the `GP`'s covariance function $k$ - the kernel of the `GP`.

### Kernels  

* The covariance matrix describes the shape of the distribution and determines the characteristics of the predicted function. $\Sigma$ is determined by evaluating the kernel/ covariance function ($k$) pairwise on all the points and acts as a similarity measure:$\newline\hspace*{24mm}k:\mathbb{R}^n\times\mathbb{R}^n\rightarrow\mathbb{R}$; $\Sigma=\text{Cov}(X,X')=k{t,t'}$
* $\Sigma_{ij}$ describes how much influence the $i^{th}$ and the $j^{th}$ point have on each other. Since $k$ describes the pairwise similarity between the function values, it controls the possible shape that a fitted function can adopt. This formulation allows the function to take on similarity measures beyond simple Euclidean distances.
* Stationary kernels such as the RBF kernel ($\sigma^2\exp(-\frac{||t-t'||^2}{2l^2})$) or the periodic kernel ($\sigma^2\exp(-\frac{2\sin^2(\pi|t-t'|/p)}{l^2})$) are invariant to translations and the covariance of two points is only dependent on their relative position. Non-stationary kernels such as the linear kernel ($\sigma_b^2+\sigma^2(t-c)(t-c')$) do not have this constraint and depend on an absolute location.
* Different kernels can describe different classes of functions, which can be used to model the desired shape of the function.

### Combining Kernels

* Kernels can be combined resulting in more specialized kernels, allowing domain experts to introduce priors to capture trends in the data.
* The covariance matrix of `GP`s has to be positive semi-definite. When choosing optimal kernel combinations, all methods that preserve this property are allowed (such as addition and multiplication).


### Prior Distribution

* Training data is the _prior_ distribution ($P_Y$). Centering the distribution ($\mu\hspace{-1mm}=\hspace{-1mm}0$) is convention. We begin with $\mu\hspace{-1mm}=\hspace{-1mm}0$ (dimensionality $|Y|\hspace{-1mm}=\hspace{-1mm}N$).
* The kernel function (dimensionality $N\times N$) determines which functions from the space of all possible functions are more probable.
* Adjusting the kernel function parameters controls the shape of the resulting functions and also varies the confidence of the predictions. For example, decreasing the variance ($\sigma$) results in sampled functions that are concentrated around the mean ($\mu$).

### Posterior Distribution

* Model the joint distribution ($P_{X,Y}\in\mathbb{R}^{|Y|+|X|}\times\mathbb{R}^{|Y|+|X|}$) and use train data ($Y$) to constrain the distribution and infer test data ($X$), i.e., the output is the conditional probability $P_{X|Y}\in\mathbb{R}^{|X|}$.
* Using conditioning, $P_{X|Y}$ can be derived from $P_{X,Y}$. Conditioning leads to derived versions of $\mu$ and $\Sigma$ : $X|Y\sim\mathcal{N}(\mu',\Sigma')$ where $\mu'\neq0$. This is because the training datapoints ($Y$) constrain the set of possible functions to those that pass through the training points.
* Further, precisely passing through $Y$ may lead to overfitting/ over-complexity, especially with noisy real-world data. We model noise/ measurement errors with the error term $\epsilon\sim\mathcal{N}(0,\psi^2)$: $X=f(Y)+\epsilon$.
* The resulting joint probability distribution is:$\newline\hspace*{12mm}P_{X,Y}=\begin{bmatrix}X\\Y\end{bmatrix}\sim\mathcal{N}(0,\Sigma)=\mathcal{N}(\begin{bmatrix}0\\0\end{bmatrix}, \begin{bmatrix}\Sigma_{XX}&\Sigma_{XY}\\\Sigma_{YX}&\Sigma_{YY}+\psi^2I\end{bmatrix})$
* We obtain obtain a prediction for our function values by sampling from the conditional distribution which is derived from conditioning on the joint distribution described above.
* Since sampling involves randomness, the prediction may not be deterministic. Marginalizing (the noise) associated with each random variable ($i$) allows us to extract the function value ($\mu_{i}'$) and uncertainty ($\sigma_{i}'=\Sigma_{ii}'$) for the $i^{th}$ test point.

### References

1. [Gaussian Processes](https://cs.stanford.edu/~rpryzant/blog/gp/gp.html) - Reid Pryzant.
2. [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/) - Jochen Görtler, Rebecca Kehlbeck, and Oliver Deussen.
3. [Gaussian Processes, not for dummies](https://thegradient.pub/gaussian-process-not-quite-for-dummies/) - Yuge Shi.

