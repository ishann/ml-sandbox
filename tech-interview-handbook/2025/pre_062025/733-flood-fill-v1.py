"""
Begin graphs with simple DFS traversal.
Run a dfs starting from [sr][sc]. Update the color.
Run dfs from all valid adjacent cells.
"""
class Solution:

    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:

        original = image[sr][sc]

        if original==color:
            return image

        m, n = len(image), len(image[0])

        def dfs(r, c):

            if image[r][c]==original:
                image[r][c]=color

            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<m and 0<=nc<n and image[nr][nc]==original:
                    image[nr][nc] = color
                    dfs(nr, nc)

        dfs(sr, sc)

        return image


    def floodFill_v0(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:

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