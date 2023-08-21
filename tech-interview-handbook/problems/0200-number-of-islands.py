"""
Problem URL: https://leetcode.com/problems/number-of-islands

Approach (DFS):
    1. Go through the entire map.
    2. If you encounter a "1", flood fill using a helper function.
        (a) Flood filler returns if area outside map.
        (b) Flood filler returns if area has already been explored, ie, not "1".
        (c) Mark as visited if both (a) and (b) conditions don't return.
        (d) Flood fill on the 4-neighbors.
    3. Increment the counter, which will be returned.

Time Complexity:
    Iterating through the map makes it O(MxN).

Space Complexity:
    We marked islands in the input matrix itself, so everything is in-place.
    O(1).
"""
class Solution:
    
    def dfs(self, grid, i, j):
        
        if i<0 or i>=len(grid) or j<0 or j>=len(grid[0]) or grid[i][j] != "1":
            return
        
        grid[i][j] = "@"
        
        for (x,y) in [(1,0),(-1,0),(0,1),(0,-1)]:
            self.dfs(grid,i+x,j+y)
        
        
    def numIslands(self, grid):
        if not grid:
            return 0
        
        print("In the beginning:\n{}".format(grid))
        count = 0
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                
                if grid[i][j]=="1":
                    count += 1
                    # Flood fill and mark "visited".
                    self.dfs(grid, i, j)
        
        print("At the end:\n{}".format(grid))
        
        return count
