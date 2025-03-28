def numIslands(self, grid: List[List[str]]) -> int:

    m, n = len(grid), len(grid[0])
    num_islands = 0

    # when new island found,
    # go to all valid connected components and sink them.
    def dfs(r, c):
        
        if r<0 or r>=m or c<0 or c>=n or grid[r][c]=="0":
            return

        grid[r][c]="0"

        for dr, dc in [[-1,0],[1,0],[0,-1],[0,1]]:
            dfs(r+dr, c+dc)

    # iterate over grid.
    for jdx in range(m):
        for idx in range(n):

            if grid[jdx][idx]=="1":
                num_islands+=1
                dfs(jdx, idx)

    return num_islands
