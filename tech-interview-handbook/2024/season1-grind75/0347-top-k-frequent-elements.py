"""
topKFrequent
    Approach:
        Create a hashmap with counts.
        Create a list of lists of length n. Each element at idx in the list will hold #occurences of a number.
        Iterate through hashmap and fill out list of lists.
        Iterate through list of lists in reverse order to generate the result.
    TC:
        Creating hashmap => O(N).
        Creating list of lists => O(N).
        Iterate through list of lists: O(N).
        => O(N).

    Space:
        Hashmap takes O(N).
        List takes O(N).
        => O(N).

topKFrequent_pythonic
    Approach:

    TC:
        Counter(nums) => O(N)
        most_common internally uses a heap => O(N.logK)
        => O(N logK)

    Space:
        Counter requires O(N).
"""
from collections import Counter
from collections import defaultdict

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:

        hashmap = defaultdict(int)
        lol = [[] for _ in range(len(nums)+1)]
        result = []

        for num in nums:
            hashmap[num] += 1

        for num, count in hashmap.items():
           lol[count].append(num)

        for idx in range(len(lol)-1, 0, -1):
            result.extend(lol[idx])
            if len(result)==k:
                return result

    def topKFrequent_pythonic(self, nums: List[int], k: int) -> List[int]:

		return [num for num, _ in Counter(nums).most_common(k)]


