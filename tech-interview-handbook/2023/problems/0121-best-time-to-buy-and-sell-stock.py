"""
Problem URL: https://leetcode.com/problems/best-time-to-buy-and-sell-stock

Approach:
    Initialize  a min_price and a max_profit.
    At each index:
        min_price = min(min_price, prices[idx])
        max_profit = max(max_profit, prices[idx]-min_price)

TC:
    Linear pass with O(1) ops => O(N).

Space:
    A few ints => O(1).
"""
class Solution:
    def maxProfit(self, prices):

        if len(prices)==1:
            return 0
        if len(prices)==2:
            return max(0, prices[1]-prices[0])
        
        min_price = min(prices[:2])
        max_profit = max(0,prices[1]-prices[0])

        for idx in range(2,len(prices)):
            min_price = min(min_price, prices[idx])
            max_profit = max(max_profit, prices[idx]-min_price)

        return max_profit

