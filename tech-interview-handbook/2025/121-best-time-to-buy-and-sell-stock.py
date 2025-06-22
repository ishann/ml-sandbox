"""
Brute force:
Try n**2 combinations of buy and then sell.
Return max(result, 0)

Optimized:
At every idx, track the min_price seen so far.
Compute the profit at current idx based on updated min_price.
If current_profit > max_profit, update.
"""
class Solution:
    
    def maxProfit(self, prices: List[int]) -> int:

        min_price = float("inf")
        max_profit = float("-inf")

        for idx in range(len(prices)):

            min_price = min(min_price, prices[idx])
            current_profit = prices[idx]-min_price
            max_profit = max(max_profit, current_profit)

        return max_profit

    def maxProfit_TLE(self, prices: List[int]) -> int:

        result = -float("inf")

        for idx in range(len(prices)-1):
            profit = max(prices[idx+1:])-prices[idx]
            if profit>result:
                result = profit

        return max(result, 0)
