# ishan_022625
"""
1 <= len(s) <= 16
with 10**1, exponential TC becomes available if required.

s cannot be empty.

TBF, we couldn't solve this at all.
Watched NeetCode a few times...
"""
class Solution:


    def isPalin(self, str_, l, r):

        substr_ = str_[l:r+1]
        return True if substr_==substr_[::-1] else False

    def partition(self, s: str) -> List[List[str]]:
        
        res = []
        part = []

        def dfs(idx):

            if idx>=len(s):
                res.append(part.copy())
                return

            for jdx in range(idx,len(s)):

                if self.isPalin(s, idx, jdx):
                    part.append(s[idx:jdx+1])
                    dfs(jdx+1)
                    part.pop()

        dfs(0)

        return res