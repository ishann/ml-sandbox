# Gaussian Processes

* Gaussian Processes (`GP`s) allow us to make predictions about data by incorporating prior knowledge. For a given set of training points, there are potentially infinitely many functions that fit the data. `GP`s offer an elegant solution by assigning a probability to each of these functions. The mean of this probability distribution represents the most probable characterization of the data. Using a probabilistic approach allows us to incorporate the confidence of the prediction into the result.
* A random variable $\mathbf{x}\in\mathbb{R}^{n}$ follows a multivariate gaussian dist in $n$ dimensions with mean $\mu\in\mathbb{R}^{n}$ and covariance $\Sigma\in\mathbb{R}^{n \times n}$ if: $\newline\hspace*{6mm}p(\mathbf{x};\mu,\Sigma) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x}-\mu)^T \Sigma^{-1} (\mathbf{x} - \mu) \right)$
* A `GP` is an extension of the multivariate gaussian to infinite dimensions. Given an input vector $x\in\mathbb{R}^{n}$, the `GP` returns a transformed vector $y\in\mathbb{R}^{n}$. Every component of $y$ represents the _probability_ of observing $x_i$ according to some Gaussian living in dimension $i$.
* Since a GP acts as a probability distribution, we can also sample new $y$ for new $x$ apart from transforming an $x$ into a $y$. Hence, if we sample the entire domain of $x$, we can obtain the function $f$ such that $f(x)=y$.

**Marginalization**: Integrate along one of the dimensions of the Gaussian to get the probability distribution over the remaining dimensions: $P(X)\hspace{-1mm}=\hspace{-1mm}\int_{-\infty}^{\infty}P(X,Y)dY$ or $P(X)\hspace{-1mm}=\hspace{-1mm}\Sigma_{Y}P(X,Y)$.

**Conditioning**: Updates the distribution over function values at new points, given observations at known points. By conditioning on observed data, `GP`s allow us to effectively incorporate priors and make informed predictions in uncertain scenarios:  
$X|Y\sim\mathcal{N}(\mu_X+\Sigma_{XY}\Sigma_{YY}^{-1}(Y-\mu_Y), \Sigma_{XX}-\Sigma_{XY}\Sigma_{YY}^{-1}\Sigma_{YX})$
$Y|X\sim\mathcal{N}(\mu_Y+\Sigma_{YX}\Sigma_{XX}^{-1}(X-\mu_X), \Sigma_{YY}-\Sigma_{YX}\Sigma_{XX}^{-1}\Sigma_{XY})$  
^^ Complex math, but note that the updated means only depend on the conditioned variable, while the covariance matrix is independent of the variable.


#### Functional view of `GP`s

  * `GP`s (which are stochastic processes) essentially are a set of random variables. Let each of these random variables have a corresponding index $i$, which represents the $i^{th}$ dimension of the $n$-dimensional multivariate distribution.
  * Goal: Learn the underlying distribution from _train data_ ($Y$).
  * `GP`s attempt to model the underlying distribution of the _test data_ ($X$), by modeling the joint distribution $P_{X,Y}$ - the span of possible function values - as a multivariate normal distribution whose dimensionality is $|X|\hspace{-1mm}+\hspace{-1mm}|Y|$).
  * Regressing on $X$ requires modeling this as Bayesian inference, i.e., update the (current) hypothesis as information ($Y$) becomes available. This conditional probability ($P_{X|Y}$) is a multivariate normal distribution as well.
  * `GP`s treat each test point as a random variable. Making predictions essentially requires sampling from the distribution; the $i^{th}$ component of the sampled vector is interpreted as the function value corresponding to the $i^{th}$ test point.

Setting up the distribution requires defining $\mu$ and $\Sigma$. In `GP`s, centering the data, i.e., $\mu=0$ is common practice. The covariance matrix ($\Sigma$) is determined by the `GP`'s covariance function $k$ - the kernel of the `GP`.

### Kernels  

* The covariance matrix describes the shape of the distribution and determines the characteristics of the predicted function.
* $\Sigma$ is determined by evaluating the kernel/ covariance function ($k$) pairwise on all the points and acts as a similarity measure:<br> $\hspace{15mm}k:\mathbb{R}^n\times\mathbb{R}^n\rightarrow\mathbb{R}$; $\Sigma=\text{Cov}(X,X')=k{t,t'}$
* $\Sigma_{ij}$ describes how much influence the $i^{th}$ and the $j^{th}$ point have on each other. Since $k$ describes the pairwise similarity between the function values, it controls the possible shape that a fitted function can adopt. This formulation allows the function to take on similarity measures beyond simple Euclidean distances.
* Stationary kernels such as the RBF kernel ($\sigma^2\exp(-\frac{||t-t'||^2}{2l^2})$) or the periodic kernel ($\sigma^2\exp(-\frac{2\sin^2(\pi|t-t'|/p)}{l^2})$) are invariant to translations and the covariance of two points is only dependent on their relative position. Non-stationary kernels such as the linear kernel ($\sigma_b^2+\sigma^2(t-c)(t-c')$) do not have this constraint and depend on an absolute location.
* Different kernels can describe different classes of functions, which can be used to model the desired shape of the function.

### Combining Kernels

* Kernels can be combined together resulting in a more specialized kernel, allowing domain experts to introduce necessary priors to capture trends in the data.
* The covariance matrix of `GP`s has to be positive semi-definite. When choosing optimal kernel combinations, all methods that preserve this property are allowed (such as addition and multiplication).


### Prior Distribution

* Training data is the _prior_ distribution ($P_Y$). Since centering the distribution ($\mu=0$) is the general convention, we begin with $\mu=0$ with dimensionality $|Y|=N$.
* The kernel function (dimensionality $N\times N$) determines which functions from the space of all possible functions are more probable.
* Adjusting the kernel function parameters controls the shape of the resulting functions and also varies the confidence of the prediction. For example, decreasing the variance ($\sigma$) results in sampled functions that are concentrated around the mean ($\mu$).

### Posterior Distribution

* Model the joint distribution ($P_{X,Y}\in\mathbb{R}^{|Y|+|X|}\times\mathbb{R}^{|Y|+|X|}$) and use train data ($Y$) to constrain the distribution and infer test data ($X$), i.e., the output is the conditional probability $P_{X|Y}\in\mathbb{R}^{|X|}$.
* Using conditioning, $P_{X|Y}$ can be derived from $P_{X,Y}$. Conditioning leads to derived versions of $\mu$ and $\Sigma$ : $X|Y\sim\mathcal{N}(\mu',\Sigma')$ where $\mu'\neq0$. This is because the training datapoints ($Y$) constrain the set of possible functions to those that pass through the training points.
* Further, precisely passing through $Y$ may lead to overfitting/ over-complexity, especially with noisy real-world data. We model noise/ measurement errors with the error term $\epsilon\sim\mathcal{N}(0,\psi^2)$: $X=f(Y)+\epsilon$.
* The resulting joing probability distribution is:$\newline$$P_{X,Y}=\begin{bmatrix}X\\Y\end{bmatrix}\sim\mathcal{N}(0,\Sigma)=\mathcal{N}(\begin{bmatrix}0\\0\end{bmatrix}, \begin{bmatrix}\Sigma_{XX}&\Sigma_{XY}\\\Sigma_{YX}&\Sigma_{YY}+\psi^2I\end{bmatrix})$.
* We obtain obtain a prediction for our function values by sampling from the conditional distribution which is derived from conditioning on the joint distribution described above.
* Since sampling involves randomness, predictions may not be deterministic. Marginalizing (the noise) associated with each random variable ($i$) allows us to extract the function value ($\mu_{i}'$) and uncertainty ($\sigma_{i}'=\Sigma_{ii}'$).

## Take II: Gaussian Processes; not for dummies.

(These notes are based on the notes taken while summarizing a talk $\implies$ It does not lend itself to compact note taking. However, the visual explanations in the [blog post](https://thegradient.pub/gaussian-process-not-quite-for-dummies/) are really helpful for building intuitions.)   

### Motivation: Non-Linear Regression

Given a few (training) datapoints $\{X_i,Y_i\}$, we try to estimate $Y_j$ at a yet unobserved $X_j$. Fitting a non-linear curve to $\{X_i,Y_i\}$ is generally the solution. This results in a single function that is considered the best fit given $\{X_i, Y_i\}$ without accounting for the possibility that future observations may not resemble the current hypothesis, i.e., there is no uncertainty associated with the current estimate.

### The World of Gaussians 

*  A random variable is k-variate normally distributed if every linear combination of its $k$ components has a univariate normal distribution: $X=(X_1, ..., X_k)^T$ is a multivariate Gaussian distribution if $Y=a_1X_1+a_2X_2+...+a_kX_k$ is normally distributed for any constant vector $a\in\mathcal{R}^k$.

#### 2D Gaussians

* New notation for sampling from 2D Gaussians: Take the oval contour graph (top-left). Choose a random point on the graph. Plot the value of $y_1$ and of $y_2$ that point on graph at index=$1$ and index=$2$: $\newline$![](https://user-images.githubusercontent.com/18204038/60428229-4e528d00-9bf0-11e9-8813-9931dd159fb8.png){width=40%}
*  The sampling operation becomes a mapping of 2D points from the oval contours (top-left) to plotting at index=$1$ and index=$2$. Since $y_1$ and $y_2$ are correlated, index graph samples can only move away from each other within some margin: $\newline$![](https://user-images.githubusercontent.com/18204038/68397330-eba76a00-016a-11ea-9950-d1dec3ee1285.gif){width=40%}
* Conditioning: Simply fix one of the endpoint on the index graph (say, fix $y_1$ to 1) and sample from $y_2$:$\newline$![](https://user-images.githubusercontent.com/18204038/68397339-ee09c400-016a-11ea-94c9-a3121f725deb.gif){width=40%}

$\vspace{20mm}$

#### 5D Gaussians

* Index Sampling: The 2D case discussed above extends to 5D Gaussians:$\newline$![](https://user-images.githubusercontent.com/18204038/68397343-f06c1e00-016a-11ea-8ad3-8203f940f495.gif){width=40%}
* Conditioning: The 2D case discussed above extends to 5D Gaussians :$\newline$![](https://user-images.githubusercontent.com/18204038/68397344-f104b480-016a-11ea-84f3-32e7485b965e.gif){width=40%}
* Non-linear regression as curve fitting is now starting to emerge from index sampling. The correlations from the covariance matrix also constrain how the index samples are able to express themselves.

#### 20D Gaussians

* Index Sampling:$\newline$![](https://user-images.githubusercontent.com/18204038/68397345-f235e180-016a-11ea-9ebb-a81bb34d2033.gif){width=40%}
* Conditioning:$\newline$![](https://user-images.githubusercontent.com/18204038/68397346-f2ce7800-016a-11ea-96d2-cce9ee1c8116.gif){width=40%}
* Non-linear regression reveals itself in the index plot. It is possible to generate a family of curves that fits the observations with one sample from the Gaussian. If we sample multiple times, we can compute the mean and variance of the fit:$\newline$![](https://user-images.githubusercontent.com/18204038/61446482-1f098300-a947-11e9-8658-18c0b6e4d50d.png){width=40%}

### Gaussian Processes

* A `GP` is a collection of random variables, any finite number of which have consistent Gaussian distributions. The `GP` is completely defined by $(1)$ a mean fncn $m(x)$ and $(2)$ a covariance fncn $K(x,x')$. The mean can be any value and the covariance matrix should be PSD: $f(x)\sim\mathcal{GP}\left(m(x), K(x,x')\right)$
* `GP`s are non-parametric while non-linear regression is parametric. But, a non-parametric model is simply a parametric model with $\infty$ parameters. 
* A parametric model explicitly defines its parameters: $y(x)=f(x)+\epsilon\sigma_y$, where $p(\epsilon)=\mathcal{N}(0,1)$ and $\sigma_y$ is Gaussian noise indicating how noisy the fit is to the actual observation.
* A `GP` prior can be placed over the above function: $p(f(x)|\theta)=\mathcal{GP}(0,K(x,x'))$, where $K(x,x')=\sigma^2\exp{(-\frac{1}{2l^2}(x-x')^2)}$.
  * **$\sigma$** (vertical scale): Describes how much span the function has, i.e., the overall function variance/ amplitude. A higher $\sigma$ allows for larger variations in function values even for nearby points, whereas a smaller $\sigma$ restricts the function to vary within a narrower range.
  * **$l$** (horizontal scale): Describes how quickly the correlation between two points drops as their distance increases. A high $l$ gives a smooth function. A low $l$ results in a wiggly function - points become uncorrelated over short distances allowing for rapid changes. 
  * Since $p(y|\theta)$ is Gaussian, we can compute the likelihood in close form. We simply maximize the likelihood of $p(y|\theta)$ under the hyperparameters using a gradient optimizer: $\arg\max_{l, \sigma^2} \log p(y|\theta)$.
* The above `GP` can also incorporate Gaussian noise ($\sigma_y$) directly into the model (sum of Gaussians is a Gaussian): $p(f(x)|\theta)=\mathcal{GP}(0,K(x,x')+I\sigma_y^2)$


### References

1. [Gaussian Processes](https://cs.stanford.edu/~rpryzant/blog/gp/gp.html) - Reid Pryzant.
2. [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/) - Jochen Görtler, Rebecca Kehlbeck, and Oliver Deussen.
3. [Gaussian Processes, not for dummies](https://thegradient.pub/gaussian-process-not-quite-for-dummies/) - Yuge Shi.

