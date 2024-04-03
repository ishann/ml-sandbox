"""
Problem URL: https://leetcode.com/problems/house-robber-ii

Approach:
    Reduce to two House Robber problems:
    1. If first house is selected, last house cannot be selected.
    2. If first house is not selected, last house can be selected.

TC:
    Two linear passes with O(1) ops => O(N)

Space:
    Maintain a dp array with N elements => O(N).
"""
class Solution:
    
    def rob(self, nums):

        N=len(nums)
        if N<=2:
            return max(nums)

        # Break the circle: Assume that the first house will _not_ be robbed.
        dp = [0]*N
        dp[0], dp[1], dp[2] = -1, nums[1], max(nums[1:3])
	  
        for idx in range(3,N):
            dp[idx] = max(dp[idx-1], dp[idx-2]+nums[idx])
	  
        case1_max = dp[-1]

        # Break the circle: Assume that the first house will _definitely_ be robbed.
        dp = [0]*N
        dp[0], dp[1], dp[-1] = nums[0], max(nums[:2]), -1
	  
        for idx in range(2,N-1):
            dp[idx] = max(dp[idx-1], dp[idx-2]+nums[idx])
	  
        case2_max = dp[-2]

        return max(case1_max, case2_max)

