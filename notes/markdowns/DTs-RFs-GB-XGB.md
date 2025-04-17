---
title: "DTs-RFs-GB-XGB"
output:
  pdf_document:
    latex_engine: pdflatex
    keep_tex: true
    toc: false
    number_sections: false
    template: default
geometry: top=.75in, bottom=.75in, left=1in, right=1in
---

# Decision Trees

`DT`s are piecewise greedy linear classifiers that can handle missing data and work with heterogenous data types.

![Simple DT for classifying survival on the Titanic.](./assets/DTRFGB_01.jpg){width=40%}

## Information Gain/ Entropy/ Gini Impurity

Decision criterions for choosing a feature to add to a DT.

### Entropy

Entropy quantifies the uncertainty or disorder in a dataset. For a dataset with classes $c_1$, $c_2$, ..., $c_k$ and probabilities $p_i$:

$\hspace*{55mm}\text{Entropy}(S)=-\Sigma_{i=1}^k p_i \log_2(p_i)$

where $p_i$ is proportion of samples in class $i$ and Entropy being $0$ indicates a "pure" node (all one class) and Entropy being $1$ indicates maximum uncertainty. 

### Information Gain

Measures reduction in entropy when data is split on a feature. Used to select the best feature to split on.

$\hspace*{25mm}\text{Information Gain}(S,A)=\text{Entropy}(S)-\Sigma_{v\in\text{Values}(A)}\frac{|S_v|}{S}\cdot\text{Entropy}(S_v)$

where $S$ is the dataset, $A$ is a feature, and $S_v$ is the subset of $S$ where $A=v$.

### Gini Impurity

Measures the probability of misclassifying a random instance.

$\hspace*{55mm}\text{Gini}(S)=1-\Sigma_{i=1}^k p_i^2$

where $p_i$ is the proportion of class $i$ in node $S$; $\text{Gini}=0$ implies a pure node.

$\vspace*{2mm}$

`NOTES`:

1. The total Entropy/ InfoGain/ Gini Impurity is a weighted sum over all child nodes.
2. Gini Impurity was used in early versions of `DT`s. Now, most algorithms stick to Entropy + InfoGain.
3. Low Gini Impurity is good for splitting. High InfoGain is good for splitting.
4. Entropy is used to compute InfoGain. They're not separate criterions.
5. Continuous features can be split on a binary threshold based on highest InfoGain.
6. We can set thresholds on when we allow splits to occur; for example $50/50$ splits are bad.
7. Missing feature values can be imputed with either the mode (for categorical features), or the mean/ median (for continuous features), or using another feature which has high correlation with this feature.

The above discussion generally applies to Classification `DT`s.

## Regression Trees

* Similar to classification trees. Split criterion is either MSE or MAE instead of InfoGain. Leaf output is mean or median instead of mode. Evaluation is MSE/ RMSE/ MAE instead of accuracy/ precision/ recall. `DT`s can include both classification and regression, though. Both splitting criterion and final evaluation will be weighted averages of classification and regression metrics.
* Splitting criterion for continuous features is a bit more annoying. Instead of checking residuals (MAE/MSE) for each threshold between two possible threshold values, consider binary search. This only works for convex residual functions.
* Regression trees where feature values (and consequently, thresholds) can take on a high number of possible values are prone to overfitting (low bias + high variance). Consider splitting based on requiring a minimum number of observed instances.
* In the case of multiple features, start with the feature sub-tree whose root gives the lowest residuals. 

![$\text{DT}_{\text{depth}=3}$ overfits. $\text{DT}_{\text{depth}=2}$ learns the major tendencies in the data and does not overfit.](./assets/DTRFGB_02.png){width=40%}

### Pruning Regression Trees - Cost Complexity Pruning

Prune sub-trees into single nodes to avoid overfitting with overly complex rules. Pruned trees will have higher residuals on training data, but are likely to generalize better.

Given $K$ candidates, $C_k$,  derived by iteratively pruning an overfit tree, a $\text{TreeScore}$ can be computed as: $\text{TreeScore}=\text{SSR}+\alpha\text{T}$ where SSR is the sum of squared residuals, T is the number of leaf nodes, and $\alpha$ is a hyperparameter tuned through cross-validation. The candidate $C_k$ with the lowest $\text{TreeScore}$ is used for predictions.

# Random Forests

`DT`s are easy to build, easy to use, and easy to interpret. But accuracy is an issue; they overfit to the training data but are not flexible when generalizing to unseen data. Random Forests (`RF`s) combine the simplicity of `DT`s with flexibility resulting in a vast improvement in generalization ability.


## Building Random Forests (BAgging)

Steps to convert a `DT` into an `RF`:

1. Create a bootstrapped dataset: randomly subsample (with replacement) from the training data.
2. Build a `DT` using a random subset of features from the bootstrapped dataset. The feature sampling occurs at each step of building the `DT`.
3. Go to #1. Repeat to generate multiple `DT`s from subsets of boths features and data.
4. During inference, each `DT` can vote (classification) for, or contribute to the mean (regression) of, the final aggregated prediction.

This method of bootstrapping to build `DT`s and aggregating predictions is known as BAgging.

`NOTES`

1. Bootstrapping (sampling with replacement) will cause some samples to never be chosen. These $\texttt{out of bag}$ samples can be used to measure the generalization ability of the `RF`.
2. The $\texttt{out of bag}$ samples can also be used to tune hyperparameters such as the probability of sampling with replacement, and the number of features sampled for building the individual `DT`s.


## Handling Missing Data

Missing data can occur both during learning and during inference.

### Missing data during learning

Make an initial guess and interatively refine it until we (hopefully) arrive at a better guess.

* Initial guess heuristic: for a missing feature value ($x_{ij}$), filter the entire by its label value ($Y_i$) and the initial guess becomes the mode (categorical)/ median (continuous) of filtered feature values ($x_{kj}$) for all $Y_i==Y_k$.
* With initial guesses filled in, build an `RF`.
* Build a proximity matrix $P$ (sample similarity matrix in the `RF` observation space): Run inference on all samples using all the `DT`s. Every sample that ends at a leaf node of a `DT` is similar to any another sample ending at the same leaf node of a `DT`. Use these proximities to populate a proximity matrix for all samples over all `DT`s and normalize it to $[0,1]$.
* For any missing feature ($x_{ij}$ with $j^{\text{th}}$ missing feature for sample $x_i$), compute weighted voting (classification)/ mean (regression) as follows:
  * Let $M_j$ be the index set of samples with observed feature $j$: $M_j=\{i\in\{1,...,N\}: x_{ij} \text{is observed}\}$.
  * The standard proximity weighted impution for missing $x_{ij}$ is: $\displaystyle \hat{x}_{ij} = \frac{\sum_{l \in M_j} P_{il}\,x_{lj}}{\sum_{l \in M_j} P_{il}}$, where $x_ {lj}$ contributes proportionally to how “close” sample  $l$ is to sample $i$ under the forest’s learned structure.
  * The aggregation becomes voting instead of summation for classification tasks.

## Similarity/ Sample Clustering 

* The proximity matrix can be used as a similarity matrix in the `RF`s output space and used for measuring similarity/ clustering/ visualization.
* The proximity matrix can be negated to form a distance matrix. This allows us to work with heterogenous data and arrive at a distance matrix in the `RF`s output space.

## Mising data during inference

* Create duplicates of the sample with missing feature and label with all possibilities.
* Run through the `RF` and use the above method to aggregate and find the most likely value of the missing feature based on the simulated values. 


References:

1. Decision and Classification Trees [statquest](https://www.youtube.com/watch?v=_L39rN6gz7Y&list=PLblh5JKOoLUKAtDViTvRGFpphEc24M-QH)
2. Random Forests [statquest](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ&list=PLblh5JKOoLUIE96dI3U7oxHaCAbZgfhHk)
3. Gradient Boosting [statquest](https://www.youtube.com/watch?v=3CC4N4z3GJc&list=PLblh5JKOoLUJjeXUvUE0maghNuY2_5fY6)
4. XGBoost [statquest](https://www.youtube.com/watch?v=OtD8wVaFm6E&list=PLblh5JKOoLULU0irPgs1SnKO6wqVjKUsQ)
5. Decision Forests [google-dev](https://developers.google.com/machine-learning/decision-forests)
6. Interpretable ML with XGBoost [medium](https://medium.com/data-science/interpretable-machine-learning-with-xgboost-9ec80d148d27)