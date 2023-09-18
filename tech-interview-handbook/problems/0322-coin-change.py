"""
Problem URL: https://leetcode.com/problems/coin-change

Problem:
    coins = denominations of the coins. we have inf coins of each denomination.
    amount = target to build using coins.
    task: return fewest NUMBER of coins.

Approach:
    optimal substructure:
        f(amount) = 1 + f(amount-min(coins))

    will have to consider all coins and pick min_{coin\in coins} dp[idx-coin]+1

TC:
    iterate for each amount.
    per amount, iterate over each coin.
    => O(amount*len(coins))
Space:
    dp of O(amount).
    => O(amount).
"""
class Solution:
    def coinChange(self, coins, amount):

        if amount==0:
            return 0

        # Smallest coin is 1. Max amount is 10**4. So, 10**4+1 is math.inf.
        dp = [0] + [10**4+1]*amount

        for idx in range(1,amount+1):
            for coin in coins:
                if idx-coin<0:
                    continue
                dp[idx] = min(dp[idx],dp[idx-coin]+1)

        return dp[-1] if dp[-1]!=(10**4+1) else -1
