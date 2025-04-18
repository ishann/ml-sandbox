"""
integers. -ve possible.

-1000 to 1000 needs 0xFFF to account for overflow.

sums without carries

carries

keep adding until carry is 0

account for -ve. show up in result being greather than max threshold.
"""
class Solution:

    def getSum(self, a: int, b: int) -> int:

        # mask to check for overflow and avoid silent wrap around.
        mask = 0xFFF
        # max will check for whether the added up sum is negative.
        max_ = mask >> 1

        # use a to store sum. use b to store carry. iterate till b>0.
        while b!=0:

            # sum without carry
            sum_ = (a^b) & mask
            # carry
            carry_ = ((a&b)<<1) & mask

            # update
            a = sum_
            b = carry_

        # check for negative result
        if a>max_:
            a = a-(mask+1)

        return a

