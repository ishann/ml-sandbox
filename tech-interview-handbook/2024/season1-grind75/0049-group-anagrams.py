"""
Approach:
    Get Counters for each strs[i].
    Insert into a defaultdict(list) with Counter being key.
    Return all the values in the defaultdict.

    NOTE: NeetCode has cleaner code than this for approximately the same high-level approach.

Time Complexity:
    Let M = strs.length.
    Let N = strs[i].length.

    Iterating over strs => O(M).
    Counter(sts_) => O(N)
    counter_to_tuple => O(26)
    => O(M.N.26) => O(M.N)

Space Complexity:
    Store possibly one unique 26-tuple for each str_ in strs in the hashmap => O(M)
"""
from collections import Counter, defaultdict

class Solution:

    def counter_to_tuple(self, counter):

        tup = [0 for _ in range(26)]

        for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz"):
            tup[idx] = counter[char]

        return tuple(tup)

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:

        hashmap = defaultdict(list)

        for str_ in strs:
            hashmap[self.counter_to_tuple(Counter(str_))].append(str_)

        return [v for k, v in hashmap.items()]
