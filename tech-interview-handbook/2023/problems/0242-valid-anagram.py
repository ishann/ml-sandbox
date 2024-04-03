"""
Problem URL: https://leetcode.com/problems/valid-anagram

Approach:
    Incrementally build a hashmap for s.
    Decrement using t.

TC:
    O(max(len(s), len(t)))

Space:
    O(num_alphabets)
"""
class Solution:
    def isAnagram(self, s, t):
        
        if len(s)!=len(t):
            return False

        smap = defaultdict(int)

        for char in s:
            smap[char] += 1

        for char in t:
            if char not in smap:
                return False
            else:
                smap[char]-=1

        return True if all(val==0 for val in smap.values()) else False