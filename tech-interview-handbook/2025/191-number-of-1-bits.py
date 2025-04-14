class Solution:
    def hammingWeight(self, n: int) -> int:
        
        if n==1:
            return 1

        num_ones = 0

        while n > 0:

            num_ones += int(n%2)
            n /= 2
            # This makes it super fast:
            # n = n >> 1

        return num_ones