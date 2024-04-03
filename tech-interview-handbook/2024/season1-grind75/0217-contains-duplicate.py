"""
Approach:
    Create a hashmap.
    Insert elements into a hashmap.
    Before inserting into hashmap, check if number exists.

Time Complexity:
    Linear parse of nums => O(N)
    Both insertion and look-up in hashmap are O(1).
    => O(N)

Space Complexity:
    Hashmap will require O(N) to store elements from nums.
"""
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:

        hashmap = set()

        for num in nums:

            if num in hashmap:
                return True

            hashmap.add(num)

        return False


    def containsDuplicate_not_very_readable(self, nums: List[int]) -> bool:

        return False if len(list(set(nums)))==len(nums) else True


