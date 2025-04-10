# Bayesian Optimization.

Bayesian Optimization can be used to optimize _any_ black-box function.

Two ways to think about sampling of an independent variable (`IV`) from a distribution to optimize finding a favorable dependent variable (`DV`):

1. **Model the Best Estimate of the DV Distribution (Active Learning)**  
  Attempt to accurately estimate the DV distribution on the IV. Sample IV such  that it provides high information about the DV distribution.

2. **Model the Optimal Dependent Variable Location (Bayesian Optimization)**  
  Find the location of the optimal DV value. Sample the IV at points that show high _promise_ about the DV.

## Active Learning (`AL`)

* Unlabeled `{x,y}` are usually inexpensive. Sampling `y` for `x` is usually expensive. Active learning minimizes labeling costs while maximizing modeling accuracy.
* Uncertainty reduction is a popular method for `AL`. This method proposes labeling the point whose model uncertainty is the highest. Often, the variance acts as a measure of uncertainty.
* Since we have access to only a few `{x,y}` pairs, a sufficiently expressive surrogate model is required to model the `IV`s the underlying function (`F`) takes at other `DV`s. Most commonly, a Gaussian Process (`GP`) is used due to its flexibility and ability to provide uncertainty estimates.
* The surrogate function (`f`) starts with a prior on `y=f(x)` and iteratively updates it according to Bayes' rule as points are sampled.     
  ![AL_fig01](https://distill.pub/2020/bayesian-optimization/images/MAB_gifs/prior2posterior.png)  
  Sample `x=.5` and find `F(x=.5)`. Update `f(x=0.5)` and reduce the uncertainty in its neighborhood. Successive samplings are performed at points with high uncertainty; usually, ones farthest from the sampled ones.

## Bayesian Optimization (`BO`)

* `AL` samples `IV`s to to determine an accurate model (`F`) for the `IV`. Estimating the underlying model is often expensive if the goal is only to find the optimal `IV`.
* `BO`: “Based on what we know so far, which point should we evaluate next?”.
* `AL` samples the most uncertain points to explore and model the underlying function (`F`). `BO` balances exploring uncertain regions across the sample space of the `DV`s while exploiting regions that are known to have close-to-optimal `IV` values using *acquisition functions* (`AF`) which are iteratively optimized to decide where to sample the `DV`s next.

**Goal**  
Find $x \in \mathbb{R}^d$ such that we are able to sample $F: \mathbb{R}^d \to \mathbb{R}$ at its optimal value.  

**Algorithm**  

1. Choose a surrogate model for modeling the true function $F$ and define its **prior**. Generally, Gaussian Processes (`GP`s) are used as surrogate models.
2. Given the set of **observations**, use Bayes rule to obtain the **posterior**.
3. Use an acquisition function $\alpha(x)$, which is a function of the posterior, to decide the next sample point: $x_t=\arg\hspace{-.5mm}\max_x\alpha(x)$.
4. ​Add sampled $\{x_t, F(x_t)\}$ to the set of **observations**. Go to #2 till convergence.


### Acquisition Functions

* Acquisition functions (`AF`s) are a critical component of `BO`. They guide the search for the next query point by balancing exploration (searching uncertain areas) and exploitation (focusing on promising areas).
* `AF`s are proxies for computationally expensive black-boxes that we want to optimize (i.e., `AF`s are significantly cheaper than the bboxes to evaluate).
* Different acquisition functions prioritize different trade-offs between exploration and exploitation.

#### $\longrightarrow$ Probability of Improvement (`PI`)
$\vspace*{1mm}$

**Intuition**

Choose points that are most likely to improve upon the best found value.

PI is focused solely on the likelihood of achieving an improvement, regardless of its magnitude. It favors areas where the model is confident about surpassing $f(x^*)$ (i.e., high $\mu(x)$ relative to $f(x^*)$ and low $\sigma(x)$, which can sometimes lead to overly exploitative behavior (which can be overcome by choosing Expected Improvement as the `AF`). Prob of Imp does not directly account for how much better the new point might be — it just tells you the chance of improvement.

**Expression**  

Probability of Improvement computes the probability that sampling at a new point $x$ will yield an improvement over the current best value  $f(x^*)$. For a `GP`, this probability is given by:  
$\hspace*{42mm}PI(x)=\Phi(\frac{\mu(x)-f(x^*)}{\sigma(x)})$

where, the term inside the $\Phi$ function represents a standardized score that indicates how many standard deviations away the predicted mean is from the best observed value. The function $\Phi$ then converts this standardized score into a probability.

**Pros || Cons**  
Simple and fast. || Can get stuck in local optima if $\sigma(x)$ is small due to being overexploitative.


### **$\longrightarrow$** Expected Improvement (EI)
$\vspace*{-8mm}$

**Intuition**

Choose points where the expected improvement over the best-so-far value is high. Expected Improvement quantifies how much, on average, you expect to improve over the current best outcome if you sample at a new point $x$.

**Expression**

If $f(x^*)$ is the best objective value observed so far, and $f(x)$ is modeled as a random variable (usually using a `GP` with mean $\mu(x)$ and variance $\sigma^2(x)$, then the improvement at $x$ is defined as:

$\hspace*{40mm}I(x)=max(f(x)-f(x^*),0)$

Since $f(x)$ is uncertain, Expected Improvment computes the expectation of $I(x)$ under the surrogate model's predictive distribution. For a Gaussian predictive distribution, the Expected Improvement is expressed in closed form as:

$\hspace*{32mm}EI(x)=(\mu(x)-f(x^*))\Phi(Z)+\sigma(x)\phi(Z)$

where $Z=\frac{\mu(x)-f(x^*)}{\sigma(x)}$, $\Phi(Z)$ is the cumulative distribution function (CDF) of the standard normal distribution, and $\phi(Z)$ is the probability density function (PDF) of the standard normal distribution.

**Pros || Cons**

Balances exploration and exploitation well. || Slightly more complex than PI but works better in practice.

EI balances exploitation (sampling where the mean $\mu(x)$ is high) and exploration (sampling where the uncertainty $\sigma(x)$ is high). The first term, $((\mu(x)-f(x^*))\Phi(Z))$, captures the expected gain when the mean is greater than the current best, weighted by how likely that is. The second term, $\sigma(x)\phi(Z)$, accounts for the contribution from uncertainty — even if the mean is not very high, a large variance might offer an opportunity for a high value. This combination makes EI a robust criterion for deciding where to sample next.

### **$\longrightarrow$** Thomson Sampling
$\vspace*{-8mm}$

Rather than calculating a direct numerical value (like `PI` and `EI`) for how *promising* a point is, Thompson Sampling draws a random function from the surrogate model's posterior distribution (typically a `GP`) and selects the next point based on the maximum of that sampled function. This process is repeated for each new evaluation.

The key advantage of Thompson Sampling is that it naturally balances exploration and exploitation: areas with high uncertainty can lead to significant variation in the sampled functions, increasing the chances of exploring those regions, while areas with high estimated performance consistently yield high values, leading to exploitation. A stochastic sampling approach often simplifies implementation and can perform robustly in practice, especially when the underlying surrogate model is well-calibrated.

### **$\longrightarrow$** Random Sampling
$\vspace*{-8mm}$

Generally not a good idea for complex models (due to high dimensionality). Surprisingly effective in simple models.

## Summary

* Principled Optimization of Expensive Functions: Bayesian optimization provides a robust framework for optimizing black-box functions where each evaluation is costly (e.g., hyperparameter tuning).
* Surrogate Modeling with Uncertainty: The approach builds a probabilistic surrogate model (often a `GP`) that estimates both the function value and the associated uncertainty, crucial for informed decision making.
* Strategic Balance via Acquisition Functions: `AF`s drive the optimization by balancing exploration (sampling uncertain regions) and exploitation (focusing on areas with high predicted performance).
* Impact of Prior Assumptions and Model Choices: The selection of kernels, noise parameters, and prior distributions plays a significant role in shaping the surrogate model’s behavior and the overall optimization effectiveness.
* Applicability to High-Cost and High-Stakes Scenarios: This method is particularly valuable for tasks where function evaluations are expensive or time-consuming, such as tuning complex ML models or designing experiments.
* Handling High-Dimensionality Challenges: While effective in many settings, Bayesian Optimization faces challenges in high-dimensional spaces, prompting the need for specialized strategies or dimensionality reduction techniques.
* Integration of Domain Knowledge: Incorporating domain expertise can improve convergence by refining the surrogate model and crafting the acquisition function more effectively.
* Iterative and Adaptive Process: The iterative nature of the approach, with continuous refinement of both the surrogate model and the acquisition strategy, is key to its robustness in dynamic optimization settings.


### References:
1. [Exploring Bayesian Optimization](https://distill.pub/2020/bayesian-optimization/)
2. [What is Bayesian Hyperparameter Optimization](https://wandb.ai/wandb_fc/articles/reports/What-Is-Bayesian-Hyperparameter-Optimization-With-Tutorial---Vmlldzo1NDQyNzcw)
3. [A Conceptual Explanation of Bayesian Hyperparameter Optimization for Machine Learning](https://medium.com/towards-data-science/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f)
