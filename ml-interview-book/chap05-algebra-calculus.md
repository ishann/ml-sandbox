# Chapter 5: Math

## Resources:
1. [CS229 Linear Algebra Primer](https://cs229.stanford.edu/section/cs229-linalg.pdf).
2. [CS229 Probability Primer](https://cs229.stanford.edu/lectures-spring2022/cs229-probability_review.pdf).

## 5.1.1 Vectors

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


## 5.1.2 Matrices

Skipped 5.1.2 for now.

## 5.1.3 Dimensionality Reduction
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