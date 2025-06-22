"""
find all the ones with 0. start BFS traversals from there.
"""
from collections import deque
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        
        m, n = len(mat), len(mat[0])

        res = [[-1]*n for _ in range(m)]

        queue = deque()

        for jdx in range(m):
            for idx in range(n):
                if mat[jdx][idx]==0:
                    res[jdx][idx]=0
                    queue.append((jdx, idx))

        directions = [(-1,0), (1,0), (0,-1), (0,1)]

        while  queue:
            r, c = queue.popleft()

            for dr, dc in directions:
                nr, nc = r+dr, c+dc

                if nr>=0 and nr<m and nc>=0 and nc<n and res[nr][nc]==-1:
                    res[nr][nc]=res[r][c]+1
                    queue.append((nr,nc))


        return res
