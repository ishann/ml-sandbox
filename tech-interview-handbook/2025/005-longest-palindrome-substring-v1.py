# ishan_022525
"""
1 <= len(s) <= 1000
empty strings are not allowed.

TC: 1e3 means that O(n**2) may be acceptable.

at each idx start from alphabet and expand in both directions until != found.
consider both odd and even palindromes starting from each idx. 

don't maintain a dp array to track indices.
instead, update the longest palin found so far in a "result" string.
"""
class Solution:
    def longestPalindrome(self, s: str) -> str:

        if len(s)==1:
            return s

        res = s[0]
        max_len = 1


        def expand(l,r):

            while l>=0 and r<len(s) and s[l]==s[r]:
                l-=1
                r+=1

            return l+1, r-1

        for idx in range(len(s)):

            odd_l, odd_r = expand(idx, idx)
            if (odd_r-odd_l+1)>=max_len:
                max_len = (odd_r-odd_l+1)
                res = s[odd_l:odd_r+1]

            even_l, even_r = expand(idx, idx+1)
            if (even_r-even_l+1)>=max_len:
                max_len = (even_r-even_l+1)
                res = s[even_l:even_r+1]

        return res

