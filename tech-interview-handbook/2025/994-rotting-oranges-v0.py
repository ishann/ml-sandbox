"""
Iterative BFS traversal.
Run through grid and figure out #fresh_oranges and locations of rotten.
Keep elapsing a minute and checking adjacent cells of rotten oranges.
"""
from collections import deque
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        
        m, n = len(grid), len(grid[0])

        rotten = deque()
        num_fresh = 0
        minutes = 0

        for jdx in range(m):
            for idx in range(n):
                if grid[jdx][idx]==1:
                    num_fresh+=1
                if grid[jdx][idx]==2:
                    rotten.append((jdx, idx))

        if num_fresh==0:
            return minutes

        directions = [(-1,0), (1,0), (0,-1), (0,1)]

        while rotten and num_fresh>0:

            for _ in range(len(rotten)):

                y, x = rotten.popleft()
                
                for dy, dx in directions:

                    ny, nx = y+dy, x+dx

                    if 0<=ny<m and 0<=nx<n and grid[ny][nx]==1:

                        grid[ny][nx]=2
                        num_fresh-=1
                        rotten.append((ny, nx))

            minutes+=1

        return minutes if num_fresh==0 else -1