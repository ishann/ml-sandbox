"""
Problem URL: https://leetcode.com/problems/find-all-duplicates-in-an-array

Approach:
    Use the array indexes as key and the sign of the array element as a val
    to create a O(1) space complexity hashmap.
TC:
    Linear pass over array with O(1) ops => O(N).
Space:
    answer is in return space, so nothing => O(1).
"""
class Solution:
    def findDuplicates(self, nums):

        answer = []

        for num in nums:
            if nums[abs(num)-1]<0:
                answer.append(abs(num))
            else:
                nums[abs(num)-1] *= -1

        return answer

