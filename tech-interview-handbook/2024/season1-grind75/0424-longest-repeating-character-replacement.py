"""
Approach:
    Use two pointers: l, r.
    Maintain a hashmap counter over current window.
    length-most_freq<=k
    Keep sliding until r<len(nums) breaks.

Time Complexity:
    Linear parse => O(N).

Space:
    Hashmap => O(26).

"""
from collections import defaultdict

class Solution:

    def characterReplacement(self, s: str, k: int) -> int:

        l=0
        counts = defaultdict(int)
        result = 0

        for r in range(len(s)):

            counts[s[r]] += 1
            while (r-l+1)-max(counts.values())>k:
                counts[s[l]]-=1
                l+=1

            result = max(result, r-l+1)

        return result


