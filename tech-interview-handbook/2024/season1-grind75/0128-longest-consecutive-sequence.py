"""
Approach:
    Convert nums to set.
    Then iterate through original list while
    using the set to make O(1) queries.
    Be careful about the book-keeping.

Time Complexity:
    Creating set is a linear parse => O(N).
    Linear parse of list of nums => O(N).
    => O(N).

Space:
    Set takes O(N).
    => O(N)
"""
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:

        longest = 0
        set_nums = set(nums)

        for num in nums:

            if num-1 not in set_nums:
                len_of_seq = 0
                end_of_seq = num
                while end_of_seq in set_nums:
                    len_of_seq+=1
                    end_of_seq+=1

                longest = max(longest, len_of_seq)

        return longest


