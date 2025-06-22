"""
Two pointers considering each character as the center from which we expand.
Make sure odd and even both work.

TC: O(n**2).

1 <= len(s) <= 1000

s = "babad"
1, b
3, bab
3, aba
1, a
1, d

s = "cbbd"
1, c
0, cb
1, b
2, bb
1, b
0, bd
1, d
"""
class Solution:
    def longestPalindrome(self, s: str) -> str:

        if len(s)==1:
            return s

        res, res_len = s[0], 1

        # iterates over each potential palindrome center.
        for idx in range(len(s)):

            # odd length
            l, r = idx, idx
            while l>=0 and r<len(s) and s[l]==s[r]:
                if r-l+1 > res_len:
                    res_len = r-l+1
                    res = s[l:r+1]
                l-=1
                r+=1

            # even length
            l, r = idx, idx+1
            while l>=0 and r<len(s) and s[l]==s[r]:
                if r-l+1 > res_len:
                    res_len = r-l+1
                    res = s[l:r+1]
                l-=1
                r+=1

        return res


