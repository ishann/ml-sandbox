"""
Net degree of judge (out-in) is N-1.
"""
class Solution:
    def findJudge(self, n: int, trust: List[List[int]]) -> int:

        num_of_trusts = [0] * n

        for [a,b] in trust:
            num_of_trusts[a-1] -= 1
            num_of_trusts[b-1] += 1

        for idx, num_of_trust in enumerate(num_of_trusts):
            if num_of_trust==(n-1):
                return idx+1

        return -1
