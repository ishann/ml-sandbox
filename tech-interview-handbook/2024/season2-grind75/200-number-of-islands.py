"""
Maintain a set of visited points in grid.
Run BFS while incrementing number of islands.
Be careful about edge conditions in BFS.
"""
from collections import deque

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:

        if not grid:
            return 0

        nrows, ncols = len(grid), len(grid[0])
        nislands = 0
        visited = set()

        def bfs(row, col):

            q = deque()
            q.append((row, col))
            visited.add((row, col))

            ngbrs = [[0,1], [0,-1], [1,0], [-1,0]]

            while q:

                r, c = q.popleft()

                for dr, dc in ngbrs:
                    cur_r, cur_c = r+dr, c+dc

                    if (cur_r in range(nrows) and
                        cur_c in range(ncols) and
                        grid[cur_r][cur_c]=="1" and
                        (cur_r, cur_c) not in visited):
                        q.append((cur_r, cur_c))
                        visited.add((cur_r, cur_c))

        for r in range(nrows):
            for c in range(ncols):

                if grid[r][c]=="1" and (r,c) not in visited:
                    bfs(r, c)
                    nislands+=1

        return nislands

