"""
Problem URL: https://leetcode.com/problems/sort-colors

Method ("Dutch Partitioning Problem"):
    Approach:
        Maintain red, white, blue.
        Invariant: Everything before white stays sorted.
    TC:
        Linear parse with O(1) ops => O(N).
    Space:
        3 ints => O(1)
Method (illegal):
    Approach:
        First pass: Count red, white, blue instances.
        Second pass: Fill up nums with the counted instances in order.
    TC:
        Two linear passes with O(1) ops => O(N).
    Space:
        3 ints => O(1)
"""
class Solution:
    def sortColors(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        """
        red, white, blue = 0, 0, len(nums)-1

        while white<=blue:
            if nums[white]==0:
                # swap red and white
                nums[red], nums[white] = nums[white], nums[red]
                # increment both red and white
                red+=1
                white+=1
            elif nums[white]==1:
                # increment white
                white+=1
            else:
                # swap white and blue
                nums[white], nums[blue] = nums[blue], nums[white]
                # decrement blue
                blue-=1


    def sortColors_illegal(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        """

        red, white, blue = 0, 0, 0

        for idx, color in enumerate(nums):
            if color==0:
                red+=1
            elif color==1:
                white+=1
            elif color==2:
                blue+=1
            else:
                print("Wrong color...!")
                break
        
        idx = 0

        while red>0:
            nums[idx] = 0
            idx+=1
            red-=1

        while white>0:
            nums[idx] = 1
            idx+=1
            white-=1

        while blue>0:
            nums[idx] = 2
            idx+=1
            blue-=1

