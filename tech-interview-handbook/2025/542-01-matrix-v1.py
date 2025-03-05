"""
Graph BFS.
Work through the motions of setting up the queue and traversing it.
"""
from collections import deque

class Solution:

    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:

        m, n = len(mat), len(mat[0])
        res = [[-1]*n for _ in range(m)]
        to_be_proc = deque()

        for jdx in range(m):
            for idx in range(n):
                if mat[jdx][idx]==0:
                    res[jdx][idx]=0
                    to_be_proc.append((jdx, idx))

        neighbors = [(-1,0), (1,0), (0,-1), (0,1)]

        while to_be_proc:

            r, c = to_be_proc.popleft()

            for dr, dc in neighbors:
                nr, nc = r+dr, c+dc

                if nr>=0 and nr<m and nc>=0 and nc<n and res[nr][nc]==-1:
                    res[nr][nc] = res[r][c]+1
                    to_be_proc.append((nr, nc))

        return res
