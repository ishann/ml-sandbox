"""
Problem URL: https://leetcode.com/problems/binary-search

Approach:
    Two pointers : left and right.
    Set mid = left+(right-left)//2
    if nums[mid]<target:
        left = mid+1 # Make sure atleast one idx gets excluded each time, to avoid INF loops.
    else:
        right=mid
TC:
    O(logN)
Space:
    left, right, mid => O(1)
"""
class Solution:
    def search(self, nums, target):

        left, right = 0, len(nums)-1
        mid = left + (right-left)//2

        while left<right:
            
            #print(nums[left], nums[right], nums[mid])
            if nums[mid]<target:
                left=mid+1
            else:
                right=mid
            mid = left+(right-left)//2
        
        return mid if nums[mid]==target else -1

        