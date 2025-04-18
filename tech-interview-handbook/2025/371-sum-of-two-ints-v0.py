"""
integers. so -ve is possible.

if integer parity, then add.
    if both -ve, then add sign in the end.
else, subtract.

to figure out mixed signs:
    mask = 0xFFF
    max_ = mask >> 1

if mixed sign exists, return result-(mask+1)
"""
class Solution:
    def getSum(self, a: int, b: int) -> int:

        mask = 0xFFF
        thresh = mask >> 1

        while b!=0:

            sum_ = (a^b) & mask
            carry_ = ((a&b)<<1) & mask
            
            a = sum_
            b = carry_

        if a > thresh:
            a = a-(mask+1)

        return a


# not accounted: (1) negatives

