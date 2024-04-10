"""
Method 1:
    Approach:
        Filter alphanumerics using str.isalnum() to create a new string.
        Compare it to its reverse.

    Time Complexity:
        Linear parse to apply str.isalnum() => O(N).
        Reversing the string using [::-1] => O(N).
        => O(N)

    Space:
        New string => O(N).
        Reversed string of new string => O(N).
        => O(N).

Method 2:
    Approach:
        Two pointers : l and r.
        Keep updating l and r until we get isAlphaNumeric and compare.
            If != return False.
        Terminate loop if !(l<r).
        return True.

    Time Complexity:
        Linear parse => O(N).

    Space:
        Nothing except two ints.
        => O(1)

"""

class Solution:

    def isAlphaNumeric(self, c) -> bool:

        isLowerAlpha = ord("a") <= ord(c) <= ord("z")
        isUpperAlpha = ord("A") <= ord(c) <= ord("Z")
        isNumeric = ord("0") <= ord(c) <= ord("9")

        return isLowerAlpha or isUpperAlpha or isNumeric

    def isPalindrome(self, s: str) -> bool:

        l, r = 0, len(s)-1

        while l<r:
            while l<r and not self.isAlphaNumeric(s[l]):
                l+=1
            while l<r and not self.isAlphaNumeric(s[r]):
                r-=1
            if s[l].lower()!=s[r].lower():
                return False
            l+=1
            r-=1

        return True

    def isPalindrome_method1(self, s: str) -> bool:

        string = [c.lower() for c in s if c.isalnum()]

        return string==string[::-1]


