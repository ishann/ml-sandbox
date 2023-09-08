"""
Problem URL: https://leetcode.com/problems/the-kth-factor-of-n/

Approach:
    Parse through all factors of n.
    Every time we find a factor, n//factor is also a factor.
TC:
    Linear pass with O(1) ops => O(sqrt(n)).
Space:
    Store factors of n in two lists and combine them => O(sqrt(n)).
"""
import math

class Solution:
    def kthFactor(self, n, k):
        
        factors, divisors = [], []

        for f in range(1, int(math.sqrt(n))+1):
            if n%f==0:
                factors.append(f)
                if f!=n//f:
                    divisors.append(n//f)
        
        all_factors = factors+divisors[::-1]

        return -1 if len(all_factors)<k else all_factors[k-1]

