"""
either 1 or 2 steps at a time.

f(n) = f(n-1) + f(n-2)

f[1] = 1
f[2] = 2

1 <= n <= 100
"""
class Solution:
    def climbStairs(self, n: int) -> int:
        
        if n<3:
            return n
        
        f = [0]*n

        f[0], f[1] = 1, 2

        for idx in range(2,n):
            f[idx] = f[idx-1] + f[idx-2]

        return f[-1]


