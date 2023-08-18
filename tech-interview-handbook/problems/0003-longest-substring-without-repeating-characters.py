"""
Problem URL: https://leetcode.com/problems/longest-substring-without-repeating-characters/description/

Approach:
    Track two pointers for left and right.
    Track seen characters and their indices in a hash map.
    Increment right from 0 to len(s).
    If s[right] seen before:
        If s[right]<left: max_len_substr = max(max_len_substr, right-left+1).
        Else: update left to s[right]+1
    Else:
        max_len_substr = max(max_len_substr, right-left+1).
    
Time Complexity:
    O(N) for linear scan with a bunch of if-else statements.

Space Complexity:
    O(C) where C is the number of unique characters used to generate strings.
"""
class Solution:
    
    def lengthOfLongestSubstring(self, s: str) -> int:

        # Hash map to track seen characters and their indices.
        seen = {}
        # Result.
        len_long_substr = 0
        # Left pointer.
        l = 0

        for idx, char in enumerate(s):

            if char in seen:
                if seen[char] < l:
                    len_long_substr = max(len_long_substr, idx-l+1)
                else:
                    l = seen[char]+1
            else:
                len_long_substr = max(len_long_substr, idx-l+1)

            seen[char] = idx

        return len_long_substr
