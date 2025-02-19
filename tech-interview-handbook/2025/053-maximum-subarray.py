# ishan_022525
"""
subarray with largest sum.
return sum, no need to maintain start/end pointers for returning.
nums[i] can be -ve.
nums cannot be empty. 1<=len(nums)
len(nums)<1e5 => need O(n) solution. At max O(n.logn). Even O(n**2) is unacceptable.

try to write a DP solution where we maintain the max sum at current idx and decide whether to
1. keep current element or start a new subarray at idx: max(nums[idx], curr+nums[idx])
2. update result : max(result, curr)
"""
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        
        if len(nums)==1:
            return nums[0]

        curr_sum = nums[0]
        result = nums[0]

        for idx in range(1,len(nums)):
            curr_sum = max(curr_sum+nums[idx], nums[idx])
            result = max(curr_sum, result)
        
        return result
