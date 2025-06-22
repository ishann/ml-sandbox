def orangesRotting(self, grid: List[List[int]]) -> int:

    m, n = len(grid), len(grid[0])

    num_fresh = 0
    num_mins = 0
    rotten = deque()

    for jdx in range(m):
        for idx in range(n):
            if grid[jdx][idx]==1:
                num_fresh+=1
            if grid[jdx][idx]==2:
                rotten.append((jdx, idx))

    if num_fresh==0:
        return num_mins

    if len(rotten)==0 and num_fresh>0:
        return -1

    while rotten and num_fresh>0:

        for _ in range(len(rotten)):

            r, c = rotten.popleft()

            for dr, dc in [[-1,0],[1,0],[0,-1],[0,1]]:

                nr, nc = r+dr, c+dc

                if nr>=0 and nr<m and nc>=0 and nc<n and grid[nr][nc]==1:
                    grid[nr][nc]=2
                    num_fresh-=1
                    rotten.append((nr,nc))

        num_mins+=1

    return num_mins if num_fresh==0 else -1
