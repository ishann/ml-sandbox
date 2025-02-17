"""
this is like climbing stairs.

at each point idx, consider the string s[:idx+1]

if f[idx-1] is a valid one-char beginning, then f[idx] += f[idx-1]
if f[idx-2:idx] is a valid two-char beginning, then f[idx] += f[idx-2] 

valid one-char is anything that is not "0"
valid two-char is anything between str(10) and str(26).

empty strings return 1

strings starting with "0" return 0.
"""
class Solution:
    def numDecodings(self, s: str) -> int:
        
        if not s:
            return 1

        if s[0]=="0":
            return 0

        n = len(s)
        f = [0] * (n+1)

        f[0] = 1
        f[1] = 1 if s[0]!="0" else 0

        for idx in range(2,n+1):

            # check valid one char.
            one_char = int(s[idx-1])
            if one_char!=0:
                f[idx] += f[idx-1]

            # check valid two char.
            two_char = int(s[idx-2:idx])
            if 10<=two_char<=26:
                f[idx] += f[idx-2]

        return f[-1]

