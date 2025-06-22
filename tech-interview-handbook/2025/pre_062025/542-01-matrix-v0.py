"""
Graph BFS.
Work through the motions of setting up the queue and traversing it.
"""
from collections import deque

class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        
        m, n = len(mat), len(mat[0])
        queue = deque()

        res = [[-1]*n for _ in range(m)]

        for r in range(m):
            for c in range(n):
                if mat[r][c]==0:
                    res[r][c] = 0
                    queue.append((r,c))

        neigh = [[-1,0],[1,0],[0,-1],[0,1]]

        while queue:

            r, c = queue.popleft()

            for dr, dc in neigh:
                visit_y, visit_x = r+dr, c+dc

                if visit_y>=0 and visit_y<m and visit_x>=0 and visit_x<n and res[visit_y][visit_x]==-1:
                    res[visit_y][visit_x] = res[r][c]+1
                    queue.append((visit_y, visit_x))

        return res