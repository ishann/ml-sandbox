# ishan_010325. Re-done after ishan_022625.
"""
1<=len(s)<=16
s cannot be empty.
len<=16 => exponential complexity solutions are on table.
"""
class Solution:

    def isPalin(self, s, l, r):
        substr = s[l:r+1]
        return True if substr==substr[::-1] else False 

    def partition(self, s: str) -> List[List[str]]:
        
        result = []
        part = []

        def dfs(idx):
            if idx>=len(s):
                result.append(part.copy())
                return
            
            for jdx in range(idx, len(s)):
                if self.isPalin(s,idx,jdx):
                    part.append(s[idx:jdx+1])
                    dfs(jdx+1)
                    part.pop()

        dfs(0)

        return result

