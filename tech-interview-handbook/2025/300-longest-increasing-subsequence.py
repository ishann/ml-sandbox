# ishan_022525
"""
strictly increasing. equality not ok.

1 <= len(nums) <= 2500
nums cannot be empty.

-1e4 <= nums[i] <= 1e4

return length not actual subseq.

iterate through the array. at each idx, maintain length assuming current included or excluded (start new).

"""

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:

        if len(nums)==1:
            return 1

        dp = [1] * len(nums)

        for idx in range(1,len(nums)):
            for jdx in range(idx):
                if nums[idx]>nums[jdx]:
                    dp[idx] = max(dp[idx], dp[jdx]+1)

        return max(dp)