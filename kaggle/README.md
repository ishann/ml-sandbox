#kaggle

Since I only kinda-sorta care about the Kaggle Leaderboard here are a few rules that might get in the way of getting a good rank:

1. Do not try to overfit to the val/ test set. Hypertuned models do not generalize well in the real-world.
2. Hyper-parameter tuning should be explored but not exploited.
3. Be aware of data leakages to learn to design better benchmarks. But do not exploit data leaks.
4. The test set distribution should not be used for filling missing values or for computing _any_ statistics.
5. Try to learn problem agnostic skills. Each dataset represents a problem domain (say, tabular data) and a task (say, classification). Focus on developing problem agnostic skills as well, apart from trying to tune numbers for a dataset.
