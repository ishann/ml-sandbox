"""
Go to each alphabet and expand until not a palindrome.
Keep incrementing counter until not a palindrome.
From each alphabet, also check if even palindromes are possible.

1 <= len(s) <= 1000
"""
class Solution:
    def countSubstrings(self, s: str) -> int:
        
        n = len(s)

        if n==1:
            return 1

        res = 0

        for idx in range(n):

            l, r = idx, idx

            # check for odd palindromes.
            while l>=0 and r<n and s[l]==s[r]:
                res+=1
                l-=1 
                r+=1

            l, r = idx, idx+1
            # check for even palindromes.
            while l>=0 and r<n and s[l]==s[r]:
                res+=1
                l-=1
                r+=1

        return res
