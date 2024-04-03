"""
Problem URL: https://leetcode.com/problems/coin-change-ii

Approach:
    Optimized KnapSack.
TC:
    Let M be len(amount).
    Let N be len(coins).
    O(MN)
Space:
    O(M)
"""
class Solution:
    def change(self, amount, coins):

        dp = [1] + [0]*amount

        for coin in coins:
            for idx in range(coin, amount+1):
                dp[idx] += dp[idx-coin]

        return dp[-1]

