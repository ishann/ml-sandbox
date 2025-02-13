"""
maximize sum of array elements without sampling two consecutive elements.

nums = [1,2,3,1]
max(1+3, 2+1)
4

f(n) = max(f(n-1), f(n-2)+nums[n])

nums = [2, 7,  9,  3,  1]
f    = [2, 7, 11, 11, 12]

1 <= len(nums) <= 100
1 <= nums_i <= 400

"""
class Solution:
    def rob(self, nums: List[int]) -> int:

        n = len(nums)

        if n==1:
            return nums[0]

        f = [0] * n

        f[0] = nums[0]
        f[1] = max(nums[0], nums[1])

        for idx in range(2,n):
            f[idx] = max(f[idx-1], f[idx-2]+nums[idx])

        return f[-1]

