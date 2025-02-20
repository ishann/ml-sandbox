# ishan_022525
"""
len(coins) < 12
amount <= 1e4
At most, we can get O(n**2) given that amount <= 1e4.
coins cannot be empty.
amount can be 0.

we can use the same coin denomination as many times as we want.

try a DP approach where we build 0 --> amount with the min number of coins.

the important observation while iterating over all coin denominations
(len(coin)<=12 makes this fine) is that dp[idx] = min(dp[idx], dp[idx-coin]+1).
"""
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        
        if amount==0:
            return 0

        dp = [float("inf")] * (amount+1)
        dp[0] = 0

        for idx in range(1, amount+1):

            for coin in coins:    
                if idx>=coin:
                    dp[idx] = min(dp[idx], dp[idx-coin]+1)

        return dp[amount] if dp[amount]!=float("inf") else -1
