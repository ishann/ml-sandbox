"""
Get all the 1s in the grid.
Have a "visited" initialized as 0s.
From each 1, start a flood fill and update visited. At end of loop, increment num_isl.
Iterate till queue non-empty.
"""
from collections import deque
class Solution:

    def numIslands_bfs(self, grid: List[List[str]]) -> int:
        
        num_islands = 0
        m, n = len(grid), len(grid[0])
        
        def bfs(r, c):
            pass

        for y in range(m):
            for x in range(n):
                if grid[y][x]=="1":
                    num_islands+=1

        return num_islands


    def numIslands(self, grid: List[List[str]]) -> int:
        
        num_islands = 0
        m, n = len(grid), len(grid[0])

        def dfs(r, c):

            if r<0 or r>=m or c<0 or c>=n:
                return
            
            if grid[r][c]=="0":
                return

            grid[r][c] = "0"
            dfs(r-1, c)
            dfs(r+1 ,c)
            dfs(r, c-1)
            dfs(r, c+1)

        for y in range(m):
            for x in range(n):

                if grid[y][x]=="1":
                    num_islands+=1
                    dfs(y,x)

        return num_islands