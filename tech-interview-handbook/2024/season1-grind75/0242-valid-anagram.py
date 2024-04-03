"""
Approach:
    Create a hashmap of letters by parsing s.
    Remove letters one at a time from hashmap while parsing through t.
    If all counts in hashmap go to 0, then we have a hashmap.

Time Complexity:
    Linear parse of s => O(lenS).
    Insertion into hashmap was O(1).

    Linear parse of t => O(lenT).
    Lookup and update is also O(1).

    Finally, linear parse of the hashmap => O(lenS).

    => O(2xlenS+lenT) = O(lenS + lenT).

Space Complexity:
    Hashmap requires O(lenS).
    Since the Q is symmetric in s and t, we can first do a min(len(s), len(t)) check to optimize space.
"""
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:

        if len(s)!=len(t):
            return False

        hashmap = {}

        for chars in s:
            if chars in hashmap:
                hashmap[chars] += 1
            else:
                hashmap[chars] = 1

        for chart in t:
            if chart in hashmap:
                hashmap[chart] -= 1
            else:
                return False

        for k, v in hashmap.items():
            if v!=0:
                return False

        return True


