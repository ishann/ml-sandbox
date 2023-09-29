import math
"""
Method 1:
    Take (m-1) steps down, and (n-1) steps right.
    => (m-1+n-1)C(m-1)

Method 2:
    Approach:
        Dynamic Programming.
        1 straight line unique path to the end of row 0 and col 0.
        for jdx in range(1,m):
            for idx in range(1,n):
                dp[jdx][idx] = dp[jdx-1][idx] + dp[jdx][idx-1]
    TC:
        O(M.N)
    Space:
        O(M.N)
"""
class Solution:
    def uniquePaths_dp(self, m: int, n: int) -> int:

        if m==1 and n==1:
            return 1

        dp = [[0]*n for _ in range(m)]

        for idx in range(1,n):
            dp[0][idx] = 1

        for jdx in range(1,m):
            dp[jdx][0] = 1

        for jdx in range(1, m):
            for idx in range(1, n):
                dp[jdx][idx] = dp[jdx-1][idx] + dp[jdx][idx-1]

        return dp[-1][-1]
        
        
    def uniquePaths_math(self, m, n):
        
        return math.comb(m+n-2, m-1)

