"""
Approach:
    Linear parse nums and create a hashmap where key is num, and value is idx.
    Do another linear parse of nums and search for target-num in hashmap.

Time Complexity:
    Two linear parses => O(lenNums).

Space Complexity:
    Hashmap requires length of nums elements => O(lenNums).

WOF:
1. Exactly one solution = early termination?
2. Cannot use same element twice. So, idx!=jdx.

"""
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:

        hashmap = {}

        for idx, num in enumerate(nums):
            hashmap[num] = idx

        for idx, num in enumerate(nums):
            if target-num in hashmap:
                jdx = hashmap[target-num]
                if idx==jdx:
                    continue
                else:
                    return [idx, jdx]
