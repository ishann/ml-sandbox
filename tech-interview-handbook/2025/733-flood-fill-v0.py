# ishan_030525
"""
Begin graphs with simple DFS traversal.
"""
class Solution:

    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:

            orig_color = image[sr][sc]

            if orig_color==color:
                return image

            def dfs(r,c):

                if r>=0 and r<len(image) and c>=0 and c<len(image[0]) and image[r][c]==orig_color:
                    image[r][c] = color
                    dfs(r-1,c)
                    dfs(r+1,c)
                    dfs(r,c-1)
                    dfs(r,c+1)

            dfs(sr,sc)

            return image