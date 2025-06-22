"""
do a DFS traversal unless image[sr][sc]==color (in which case return).
"""
class Solution:
    def floodFill(self,
                 image: List[List[int]],
                 sr: int, sc: int,
                 color: int) -> List[List[int]]:
        
        original = image[sr][sc]
        if original==color:
            return image

        m, n = len(image), len(image[0])

        def dfs(r, c):

            if r>=0 and r<m and c>=0 and c<n and image[r][c]==original:
                    
                    image[r][c]=color

                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        dfs(r+dr, c+dc)

        dfs(sr, sc)

        return image
