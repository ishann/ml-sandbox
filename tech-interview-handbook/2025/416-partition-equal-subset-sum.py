# ishan_022525
"""
to get two subsets, we need to sample a subset of elements that can add up to sum(nums)//2.
total = sum(nums)

if total//2!=total/2, then reject.

numbers can't be negative.
nums can't be empty.

[1,5,11,15]
total = 32
find a subset that adds to 16.

at first glance, this appears to be a 0-1 knapsack problem.
try to write a DP solution.

for num in nums:
    for idx in range(total//2, num-1, -1)
        f[idx] = f[idx] or f[idx-num]
"""
class Solution:
    def canPartition(self, nums: List[int]) -> bool:

        total = sum(nums)

        if total%2!=0:
            return False

        target = total//2

        dp = [False] * (target+1)
        dp[0] = True

        for num in nums:
            for idx in range(target, num-1, -1):
                dp[idx] = dp[idx] or dp[idx-num]

        return dp[target]
        