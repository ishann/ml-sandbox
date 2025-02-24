# ishan_022625
"""
1 <= len(s) <= 100

s cannot be empty.
100 is quite permissive so O(n**2) and even O(n**3) may work.

12

1,2
12

226
2,26
2,2,6
22,6

go left to right. store at each idx, how many unique ways possible.
f[idx] will only depend on f[idx-1] for possible one chars and
f[idx-2:idx-1] for possible one chars before it.
"""
class Solution:
    def numDecodings(self, s: str) -> int:
        
        n = len(s)

        if n==1 and int(s[0])!=0:
            return 1

        if int(s[0])==0:
            return 0

        dp = [0] * (n+1)

        dp[0] = 1
        dp[1] = 1 if int(s[0]!=0) else 0

        for idx in range(2,n+1):

            # check for one char before.
            one_char = int(s[idx-1])
            if one_char!=0:
                dp[idx] += dp[idx-1]

            # check for two chars before.
            two_char = int(s[idx-2:idx])
            if two_char>=10 and two_char<=26:
                dp[idx] += dp[idx-2]

        return dp[n]