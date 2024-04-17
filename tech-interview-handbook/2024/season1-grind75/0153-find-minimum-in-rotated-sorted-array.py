"""
0:[0,1,2,4,5,6,7]
4:[4,5,6,7,0,1,2]
2:[6,7,0,1,2,4,5]

Approach:
    Binary Search.
    Condition: if nums[mid]>nums[right]: left=mid+1
               else: right=mid

TC:
    Binary search takes O(logN).

Space:
    left, right => O(1).
"""
class Solution:
    def findMin(self, nums: List[int]) -> int:

        left, right = 0, len(nums)-1

        while left<right:

            mid = left + (right-left)//2

            if nums[mid]>nums[right]:
                left = mid+1
            else:
                right = mid

        return nums[left]


