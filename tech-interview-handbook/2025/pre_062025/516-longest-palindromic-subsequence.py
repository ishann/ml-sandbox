"""
One more backtracking.
Last one.
No more.
"""
class Solution:

    def longestPalindromeSubseq_TLE_no_memo(self, s: str) -> int:

        def backtrack(l, r):

            if l==r:
                return 1

            if l>r:
                return 0

            if s[l]==s[r]:
                return 2+backtrack(l+1, r-1)
            else:
                return max(backtrack(l+1, r), backtrack(l, r-1))

        return backtrack(0, len(s)-1)


    def longestPalindromeSubseq(self, s: str) -> int:
        
        memo = {}
        
        def backtrack(l,r):

            if l==r:
                return 1

            if l>r:
                return 0

            if (l,r) in memo:
                return memo[(l,r)]

            if s[l]==s[r]:
                memo[(l,r)] = 2+backtrack(l+1, r-1)
            else:
                reject_left = backtrack(l+1, r)
                reject_right = backtrack(l, r-1)
                memo[(l,r)] = max(reject_left, reject_right)

            return memo[(l,r)]

        return backtrack(0,len(s)-1)

