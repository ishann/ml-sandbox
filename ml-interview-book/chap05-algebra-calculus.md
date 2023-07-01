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
