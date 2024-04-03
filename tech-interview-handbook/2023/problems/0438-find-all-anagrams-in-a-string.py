"""
Problem URL: https://leetcode.com/problems/find-all-anagrams-in-a-string

Approach:
    Create hashmap from p: pmap.
    Run sliding window over s and do the following:
        If s[idx] in pmap: increment pmap[s[idx]]
        If s[idx+plen] in pmap: decrement pmap[s[idx+plen]]
        If all counters in hashmap are 0, we found an anagram.
TC:
    O(len(s)) * O(pmap) = O(N) * O(26) = O(N) * O(1) = O(N)
Space Complexity:
    O(26) = O(1)
"""
class Solution:
    def findAnagrams(self, s, p):

        slen, plen = len(s), len(p)

        if plen>slen:
            return []

        pmap = defaultdict(int)
        res = []

        for c in p:
            pmap[c]+=1

        for idx in range(plen-1):
            if s[idx] in pmap:
                pmap[s[idx]] -= 1

        for idx in range(-1,slen-plen+1):
            if idx>-1 and s[idx] in pmap:
                pmap[s[idx]] += 1
            if idx+plen<slen and s[idx+plen] in pmap:
                pmap[s[idx+plen]] -= 1

            if all(v==0 for v in pmap.values()):
                res.append(idx+1)

        return res
