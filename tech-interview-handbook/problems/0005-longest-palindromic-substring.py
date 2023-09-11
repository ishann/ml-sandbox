"""
Problem URL: https://leetcode.com/problems/longest-palindromic-substring

Approach:
    Consider each index.
    Spread the index to right until we see the same character.
    Now, spread two pointers in both directions.
    Track the length.
    Return type is string itself, not just the length, so
    track either [beg, end] or [beg, length].

TC:
    Linear pass with linear scan => O(N**2)

Space:
    A few indexes => O(1).
"""
class Solution:
    def longestPalindrome(self, s):
        
        N = len(s)
        if N<2:
            return s

        max_pal_len, max_pal_begend = 1, [0,0]

        for idx in range(N):

            r = idx

            # Spread the index to right until we see the same character.
            # For cases such as "c<bb>d" or "a<bbb>ad".
            while r<N and s[r]==s[idx]:
                r+=1

            # Now, spread by 1 step towards both left and right.
            l = idx-1
            while l>=0 and r<N and s[l]==s[r]:
                l-=1
                r+=1
            
            # (r-1)-(l+1)+1 = r-l-1
            if r-l-1>max_pal_len:
                max_pal_len = r-l-1
                max_pal_begend = [l+1,r-1]
        
        beg_, end_ = max_pal_begend
        
        return s[beg_:end_+1]

