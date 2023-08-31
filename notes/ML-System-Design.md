# ML System Design

## High-level Ideas
1. First, build _simple_ pipelines _quickly_ in a _modular_ fashion to serve as _effective_ baselines. Then, iterate.
2. We train once. We deploy and serve often. A simple and effective system may be preferable to a complicated SoA system.
3. Define the (business) problem and how it maps to a system that may or may not use ML to solve the problem. Identify what *success* looks like in the context of relevant metrics. If we don’t understand how the model fits into the business problem, it doesn’t matter how amazing the ML model is.
4. Always work backwards from the end-user. Good UX makes up for a lot of limitations of the ML model.
5. Evaluation is super-critical. If we do not have a way to measure performance, we cannot improve the model. If we cannot improve the model, we cannot improve the system. If we cannot improve the system, we cannot improve the business. If we cannot improve the business, we cannot help the user. If we cannot help the user, we do not build it.
6. Don't get carried away by the modeling bits. Data pre-processing/ post-processing, logging, evaluation metrics, inference computational and run-time performance are quite important.    
7. Break things down. ML systems are complex systems. Break down the complex system into concrete sub-systems that interact with each other. Break down each sub-system into ML and non-ML services. Consider further breaking down the ML service.
8. Always measure everything. At the minimum, consider logging every service's run-time, computational, and memory requirements. Not all errors are equal; measure *all* kinds of errors.
9. Account for model drift. Consider:
   1. Detecting drift between streaming distribution and train distribution.
   2. Triggers to retrain.
   3. Evaluating quality of retrained models.
10. Performance constraints
    1.  Trade off speed and quality of predictions.
    2.  Precision vs. recall.
    3.  Are errors equal? False Negative vs. False Positive?
11. Personalization: Do we need one model for all users, or for user segments, or for individual users? At what abstraction does this personalization occur?
12. Model the problem in a way that allows us to train once/ a few times. Re-training should not be _required_ everytime the streaming distribution changes.
13. More data is a boring answer (diminishing returns). Instead, ask what _complementary_ data or new features can be added to the training distribution.

## Metrics
References: [[1]](https://medium.com/usf-msds/choosing-the-right-metric-for-machine-learning-models-part-1-a99d7d7414e4), [[2]](https://medium.com/usf-msds/choosing-the-right-metric-for-evaluating-machine-learning-models-part-2-86d5649a5428)

### Regression

1. Root Mean Squared Error: $\sqrt{\frac{1}{n}\Sigma_{i=1}^{n}{(y-\hat{y})^2}}$

2. Mean Absolute Error: $\frac{1}{n}\Sigma_{i=1}^{n}{|y-\hat{y}|}$

3. $R^2 =  1-\frac{\Sigma_{i=1}^{n}(y_i-\hat{y_i})^2}{\Sigma_{i=1}^{n}(y_i-\bar{y_i})^2}$
   
4. $R_{adj}^2=1-\frac{(1-R^2)(n-1)}{n-k-1}$ (where $k$ is the num of independent predictors)

Minimizing the squared error ($L_2$) results in finding its mean, and minimizing the absolute error ($L_1$) results in finding its median. Minimizing RMSE is easier to solve for while minimizing MAE is more robust to outliers.    
$R_{adj}^2$ accounts for the number of independent predictors (IVs) and is especially useful when the number of such IVs is large.    


### Classification

1. Accuracy: Ratio of correct predictions to all predictions = $\frac{\text{TP}+\text{TN}}{\text{P}+\text{N}}$

3. Precision: Ratio of correctly positive predictions to all positive predictions = $\frac{\text{TP}}{\text{TP}+\text{FP}}$

4. Recall/ Sensitivity/ TPR : Ratio of correct positive predictions to all positives = $\frac{\text{TP}}{\text{TP}+\text{FN}}$

5. Specificity/ TNR: Ratio of correct negative predictions to all negatives = $\frac{\text{TN}}{\text{TN}+\text{FP}}$

6. Type I Error/ FPR: Ratio of wrong positive predictions to all negatives = $\frac{\text{FP}}{\text{TN}+\text{FP}}$

7. Type II Error/ FNR: Ratio of wrong negative predictions to all positives = $\frac{\text{FN}}{\text{TP}+\text{FN}}$

8. F1 Score: Harmonic mean of precision and recall = $\frac{2}{\frac{1}{\text{Precision}}+\frac{1}{\text{Recall}}}$.

9.  ROC : Threshold independent TPR vs FPR curve.    
AUC interpretation: In binary classification, the probability that a $+$ outranks a $-$.

1.  BCE : Binary Cross-Entropy = -$\frac{1}{n}\Sigma_{i=1}^{n}{y_i\log{\hat{y_i}}+(1-y_i)\log{(1-\hat{y_i})}}$

BCE considers absolute probabilistic correctness.    
AUC considers threshold independent predictive correctness (and can also be nice to minority classes).    
F1 considers threshold dependent predictive correctness.


## Debugging ML models
1. Do not underestimate the value of staring at input data, code, and neuralnet intermediate states. When in doubt, `ipdb` through the entire pipeline. [[cf. See the world as your agent does](https://openai.com/research/openai-baselines-dqn)]
2. Log _everything_. Network I/O is not enough. Log intermediate states, high-confidence errors, optimizer state changes, etc. Log system performance such as disk read times, inference runtimes, GPU/ RAM usage, etc.
   1. "... you should not just track loss values, but also optimizer information like update norm, gradient norm, norm of momentum term, angle between them etc. ... allow you to distinguish between whether the model parameters converged at a local minimum, or if they are slowly traversing a flat region in the loss surface. In the latter case, dynamically increasing the learning rate can help." (cf. [r/ML](https://www.reddit.com/r/MachineLearning/comments/ouiegi/d_sudden_drop_in_loss_after_hours_of_no/))
3. Poor predictive performance does not imply the presence of a bug. Instead, to debug poor performance in a model, you investigate a broader range of causes than you would in traditional programming. [[cf. developers.google.com](https://developers.google.com/machine-learning/testing-debugging/common/overview)]
   1. Do features lack predictive power?
   2. Are hyperparameters sub-optimal?
   3. Does data contains errors/ anomalies?
4. Borrow ideas from software-dev
   1. Explain the problem to a rubber duck, ChatGPT, or a sympathetic colleague.
   2. Unit tests: data processing/ training/ inference.
   3. Integration tests: end-to-end pipeline.
   4. Monitor data drift/ model drift.
   5. Log performance, including second-order metrics such as change in prediction confidence.
5. (Deliberately) overfit to a tiny subset of the data.
6. Start with simple baselines that provide reasonably poor performance. Then, iterate.
7. Check for overfitting
   1. Regularization: $\lambda \Sigma_i |W_i|_p$.
   2. Get more training data. Or atleast apply data augmentation.
   3. Dropout.
   4. Early stopping.
   5. Use adaptive optimizers that adjust learning rate/ weight decay.
8. Maintain a project-agnostic [`mltest`](https://github.com/Thenerdstation/mltest) repository that can be incorporated into and modified for each new project.
9.  Avoid multi-machine distributed training when setting up baselines. SSGD is slow; ASGD can introduce numerical issues that should be dealt with during model deployment, not model development. When we must, consider ideas from "[Training Neural Nets on Larger Batches: Practical Tips for 1-GPU, Multi-GPU & Distributed setups](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255)".
10.  "A fast-and-furious approach to training neural networks does not work and only leads to suffering ... can be mitigated by being thorough, defensive, paranoid, and obsessed with visualizations of basically every possible thing. ... qualities that correlate most strongly to success in deep learning are patience and attention to detail." ([cf. Karpathy](https://karpathy.github.io/2019/04/25/recipe/))    
    1. Become one with the data.    
    2. Set up the end-to-end training/evaluation skeleton + get simple baselines.    
    3. Overfit to tiny data.    
    4. Regularize.    
    5. Tune hyperparameters methodlogically.    
    6. Persevere.    

## Framework
[Patrick Halina](http://patrickhalina.com/posts/ml-systems-design-interview-guide/).
Other links:
[link1](https://towardsdatascience.com/how-to-answer-any-machine-learning-system-design-interview-question-a98656bb7ff0),
[link2](https://towardsdatascience.com/what-is-machine-learning-system-design-interview-and-how-to-prepare-for-it-537d1271d754),
[link3](https://www.educative.io/blog/cracking-machine-learning-interview-system-design),
[link4](https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md#2-ml-system-design-sample-questions).


### Design Process

#### Product Objectives and Requirements Clarification

#### High Level Design

#### Data Brainstorming and Feature Engineering

#### Infrastructure

#### Model Development

#### Leveling

#### 


References:
1. [Machine Learning System Design](https://huyenchip.com/machine-learning-systems-design/toc.html) by [Chip Huyen](https://huyenchip.com).
2. [Debugging ML Model Training](https://neptune.ai/blog/debugging-deep-learning-model-training) by [neptune.ai](https://neptune.ai).
3. [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) by [Andrej Karpathy](https://karpathy.github.io)
4. [ML Systems Design Interview Guide]() by 
