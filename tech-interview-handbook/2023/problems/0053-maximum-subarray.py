"""
Problem URL: https://leetcode.com/problems/maximum-subarray

Approach:
    Consider dynamic programming.
    Maintain a list, dp, which holds the max sum possible while including element at idx.
    Also, maintain a variable that will be max thus far. 
TC:
    Linear pass with O(1) ops => O(N)
Space:
    The dp list is O(N) => O(N)
"""
class Solution:
    def maxSubArray(self, nums):
        
        if len(nums)==1:
            return nums[0]

        max_ = nums[0]
        dp = [0] * len(nums)
        dp[0] = nums[0]

        for idx in range(1,len(nums)):
            dp[idx] = max(dp[idx-1]+nums[idx], nums[idx]) # type: ignore
            max_ = max(max_, dp[idx])

        return max_

