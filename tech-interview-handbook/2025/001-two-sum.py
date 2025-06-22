"""
Maintain hashmap from val to idx.
Search in hashmap for target-num
"""
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        
        hashmap = {}

        for idx in range(len(nums)):
            if target-nums[idx] in hashmap:
                return [hashmap[target-nums[idx]], idx]
            hashmap[nums[idx]]=idx
