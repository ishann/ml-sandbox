"""
only 1 ... 9 can be used.
each used once.

nums={1...9}Ck such that sum(nums)=n.

2<=k<=9
1<=n<=60 (actually 1<=n<=45 since only 1...9 can be used once)

get to n by iteratively doing DFS and building to it.
when a path satisfies sum=n, we add to result.
when constraints violated (sum>n, candidates>k), we backtrack.
"""
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        
        if n>45:
            return []

        res = []

        def backtrack(start, path, target):

            if len(path)==k:
                if target==0:
                    res.append(path[:])
                return

            for num in range(start, 10):
                if num>target:
                    break

                path.append(num)
                
                backtrack(num+1, path, target-num)
                
                path.pop()

        backtrack(1, [], n)

        return res