Reference: [ML System Design by ByteByteGo](https://bytebytego.com/courses/machine-learning-system-design-interview).

![Components of a production ready ML system.](./assets/ch1-01.png)

## Introduction and Overview

ML system design interviews expect us to answer open-ended questions. There is no single correct answer. We are evaluated on our thought process, our in-depth understanding of various ML topics, our ability to design an end-to-end system, and our design choices based on the trade-offs of various options.

Unstructured answers make the flow difficult to follow and derail open-ended discussions. It is important to follow a framework:

![ML system design steps.](./assets/ch1-02.png)

* Clarify the objective to help shape primary requirements. Transform the objective/ requirements into a tangible ML task.
* Data preparation is often overlooked. Don't be an idiot. Think through what will be fed to the model, both in terms of `X` and `Y`.
* Model development and evaluation must be an continuously iterative process. To `MeasureEverything` is important; ideally evaluation (and non-ML baselines) must be decided in the early stages of model development.
* Continuous monitoring after serving is essential to ensure that the objectives are being met and that the data/ concept does not drift.


### Clarify Requirements

Ask clarifying questions; asking questions exhibits attention to detail and intention. Attempt to understand the exact requirements and what the user (the interviewer) cares about:

* **Business objective**: Increase revenue? Increase `#`users? Increase engagement per user?

* **Features the system needs to support**: Move towards tangible features. Clarify if the features are essential or good-to-haves. Discuss feasibility while keeping a realistic ML system in mind. For example, a recommendation system might allow users to “like” or “dislike” recommendations, as those interactions could be used to label training data.

* **Data**: Data sources? How large is a datum and the dataset? Is it structured or unstructured? Do we have labels and is there a possibility for soft labels?

* **Constraints**: How much computing power is available (during inference; during learning we assume $\infty$ resources)? Is it a cloud-based system, or should the system work on a device? Is the model expected to improve automatically over time?

* **Scale of the system**: How many users do we have? How many items, such as videos, are we dealing with? What’s the rate of growth of these metrics?

* **Performance**: How fast must inference be? Is a real-time solution expected? Does accuracy have more priority or latency?

* **FairML**: Do Fairness, Accountability, Transparency, and Explainability matter?


### Frame the Problem as an ML Task

Effective problem framing is essential for translating a business objective into a clear ML task. This involves `(1)` ensuring that ML is necessary (though an ML system design interview probably requires an ML system$...$), `(2)` defining a precise ML objective that models can address (e.g., maximizing click-through rate instead of merely “increasing sales”), `(3)` specifying the system’s inputs and outputs as a blackbox — especially when multiple models are involved, and `(4)` choosing the appropriate ML paradigm (e.g., supervised learning for most cases, with further distinctions between binary/ multiclass classification or regression), ensuring the task aligns with the nature of available data and the intended outcome.

#### Define the ML Objective

Possible business objective: increase sales by `20%`. But, we cannot optimize over sales directly. For an ML system to solve a task, we must translate the business objective into a well-defined ML objective (say, to improve recommendations so users buy more items). We could also increase sales by increasing the `#`users by `20%`, but that might require moving away from improving recommendations towards, say, a better ad-serving system. Increasing sales can only be measured on the field and that is not a good ML objective for learning and evaluating the ML system.

A *good ML objective* is one that ML models can solve and be evaluated upon. 

#### Specify the System's I/O

As a black-box what are `X` and `Y`? In some cases, the system may be complex and we may need to specify the I/O of each component ML model. There may be more than one way to specify the I/O and these decisions matter for how we make downstream decisions.

![Different ways to specify the model's I/O.](./assets/ch1-04.png){width=80%}


