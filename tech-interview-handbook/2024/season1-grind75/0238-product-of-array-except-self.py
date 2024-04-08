"""
Approach:
    I want two arrays:
        one with products upto that idx.
        another with products starting after that idx.
    These two arrays will help us get the per-idx results on the 3rd pass.

Time Complexity:
    Linear parse to create "before" => O(N).
    Linear parse to create "after" => O(N).
    Linear parse to get the result => O(N).
    => O(N).

Space:
    "before" and "after" take O(N) each.
    => O(N).

Scratch Space:

    nums = [1,2,3,4]
    N = 4

    before = [1,1,1,1]
    after = [1,1,1,1]
    result = [1,1,1,1]

    before = [ 1,  1, 2, 6]
    after  = [24, 12, 4, 1]

"""
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:

        N = len(nums)
        before = [1] * N
        after = [1] * N
        result = [1] * N

        bef = 1
        for idx in range(N):
            before[idx] = bef
            bef *= nums[idx]

        aft = 1
        for idx in range(N-1,-1,-1):
            after[idx] = aft
            aft *= nums[idx]

        for idx in range(N):
            result[idx] = before[idx]*after[idx]

        return result
