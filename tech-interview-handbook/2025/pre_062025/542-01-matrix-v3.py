"""
1. parse mat and find where 0s are. mark their dists to be 0. put 0 locations in a deque.
2. check neighbors or deque elements. if they havent been visited update their dists to one more than deque element. add them to deque.
"""
from collections import deque
def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
    
    m, n = len(mat), len(mat[0])
    to_proc = deque()
    dists = [[-1]*n for _ in range(m)]

    for jdx in range(m):
        for idx in range(n):
            if mat[jdx][idx]==0:
                dists[jdx][idx]=0
                to_proc.append((jdx, idx))

    while to_proc:

        r, c = to_proc.popleft()

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r+dr, c+dc
            if nr>=0 and nr<m and nc>=0 and nc<n and dists[nr][nc]==-1:
                dists[nr][nc] = dists[r][c]+1
                to_proc.append((nr, nc))

    return dists

        
