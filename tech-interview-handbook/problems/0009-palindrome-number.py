"""
1. Check if negative number.
2. Convert to string. Then, check from both ends til len(s)//2.
"""
class Solution:
    def isPalindrome(self, x: int) -> bool:

        if x<0:
            return False

        s = str(x)
        n = len(s)

        for idx in range(n//2):
            if s[idx]!=s[n-1-idx]:
                return False

        return True


