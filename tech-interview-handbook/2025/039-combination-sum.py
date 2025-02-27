"""
Tried Combination Sum III right before.
Trying this now to make sure the concept is clear.
"""
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        
        res = []

        def backtrack(start, path, target):

            if target==0:
                res.append(path[:])
                return

            if target<0:
                return

            for idx in range(start, len(candidates)):

                path.append(candidates[idx])

                backtrack(idx, path, target-candidates[idx])

                path.pop()

        backtrack(0, [], target)

        return res