"""
Problem URL: https://leetcode.com/problems/optimal-partition-of-string

Approach:
    Start parsing the string and store each seen character in a set.
    Every time a duplicate character is encountered:
        Update counter.
        Reset the set.

TC:
    Linear parse with O(1) ops => O(N)

Space:
    Set which may contain at max K chars => O(K).
    K is number of unique alphabets.
    K<=N
"""
class Solution:
    def partitionString(self, s: str) -> int:
        
        seen = set()
        counter = 1

        for idx, c in enumerate(s):

            if c not in seen:
                seen.add(c)
            else:
                counter+=1
                seen = set()
                seen.add(c)

        return counter

