"""
Go to each alphabet and expand from center.
Check for both even and odd palindromes.
"""
class Solution:
    def countSubstrings(self, s: str) -> int:
        
        n = len(s)
        if n==1:
            return 1

        res = 0

        for idx in range(n):

            l, r = idx, idx
            while l>=0 and r<n and s[l]==s[r]:
                res+=1
                l-=1
                r+=1

            l, r = idx, idx+1
            while l>=0 and r<n and s[l]==s[r]:
                res+=1
                l-=1
                r+=1
        
        return res