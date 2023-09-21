"""
Problem URL: https://leetcode.com/problems/partition-equal-subset-sum/

Approach:
    At idx x,
        can you build amount x using elements from num.
    
    iterate on this from top to bottom.
    but results will fill up bottom to top.

    dp = [True] + [False]*(half_sum)
TC:
    Let N = len(nums)
    Let T = sum(nums)
    O(N.T//2) => O(NT)
Space:
    O(T//2) => O(T)
"""
class Solution:

    def canPartition(self, nums):

        if len(nums)==1:
            return False

        total = sum(nums)
        half_total = total//2
        
        if total%2==1:
            return False

        dp = [True] + [False] * half_total

        for num in nums:
            for amount in range(half_total, num-1, -1):
                dp[amount] = dp[amount] | dp[amount-num]

        return dp[half_total]

