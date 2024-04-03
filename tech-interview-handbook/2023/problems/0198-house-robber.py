"""
Problem URL: https://leetcode.com/problems/house-robber/

Approach:
    Consider dynamic programming.
    dp = [0]*len(nums)
    dp[idx] = max(dp[idx-1], dp[idx-2]+nums[idx])
TC:
    Linear parse of nums with O(1) ops => O(N)
Space:
    Maintain dp, with N elements => O(N)
    Should be possible to optimize this, since
    we only need dp[idx-1] and dp[idx-2] to compute dp[idx]
"""
class Solution:
    
    def rob(self, nums):
        N = len(nums)
    
        # If upto 2 houses.
        if N<2:
            return max(nums)
    
        dp = [0] * N
        dp[0], dp[1] = nums[0], max(nums[:2])

        for idx in range(2,N):
            dp[idx] = max(dp[idx-1], nums[idx]+dp[idx-2])

        print(dp)
        return dp[-1]    
    
