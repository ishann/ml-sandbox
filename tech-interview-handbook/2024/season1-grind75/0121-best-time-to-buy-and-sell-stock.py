"""
Approach:
    Two pointers with sliding window, initialized as: l, r=0, 1.
    while r<len(prices):
        update max_profit.
        if prices_l > prices_r:
            l=r
        r+=1

Time Complexity:
    Linear parse of prices => O(N).

Space:
    Two int pointers => O(1).

"""
class Solution:
    def maxProfit(self, prices: List[int]) -> int:

        max_profit = 0
        l, r = 0, 1

        while r<len(prices):
            if prices[l]<prices[r]:
                max_profit = max(max_profit, prices[r]-prices[l])
            else:
                l=r
            r+=1

        return max_profit


