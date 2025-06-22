# ishan_022525
"""
robot can only move down or right. no up. no left.

--> version 0
1<=m,n<=100
Out of m-1+n-1 steps to be taken, select m-1 and mark as row steps and the remaining n-1 as col steps: (m-1+n-1)C(m-1).

--> version 1
factorials are too slow when m,n become slightly bigger numbers.

create a 2D DP. dp[i][j] = dp[i-1][j] + dp[i][j-1]

return dp[-1][-1]
"""
from math import comb
class Solution:

    def uniquePaths(self, m: int, n: int) -> int:

        if m==1 and n==1:
            return 1

        dp = [[0]*n for _ in range(m)]

        for idx in range(m):
            dp[idx][0] = 1

        for jdx in range(n):
            dp[0][jdx] = 1

        for idx in range(1,m):
            for jdx in range(1,n):

                dp[idx][jdx] = dp[idx-1][jdx] + dp[idx][jdx-1]

        return dp[-1][-1]


    def uniquePaths_v0(self, m: int, n: int) -> int:
        
        return comb(m-1+n-1,m-1)