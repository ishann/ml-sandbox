"""
Problem URL: https://leetcode.com/problems/detect-cycles-in-2d-grid

Approach:
    Run DFS util from each node. Return True if cycle detected. Return False if full grid parsed.
    DFS util:
        Maintain a seen node, which signals that a cycle was found.
        Check legitimacy of each 4-neighbors and run DFS from each legit neighbor.
TC:
    Each node will be visited once at the most => O(MxN)
Space:
    seen can hold all nodes at the most => O(MxN)
"""
class Solution:
    def containsCycle(self, grid):
        def dfs(node, parent):
            """
            Run DFS starting from a node, while being aware of parent's location.
            node: where we are right now.
            parent: where we coming from.
            """
            if node in seen:
                return True
            seen.add(node)
            ny, nx = node
            children = []
            nghbrs = [(ny-1,nx), (ny+1,nx), (ny,nx-1), (ny,nx+1)]

            for oy, ox in nghbrs:
                """1. within the grid.
                   2. transition only if same value
                   3. transition, except to the parent"""
                if oy<M and oy>=0 and ox<N and ox>=0 and grid[oy][ox]==grid[ny][nx] and (oy, ox)!=parent:
                    children.append((oy, ox))

            for child in children:
                if dfs(child, node):
                    return True
            
            return False


        M, N = len(grid), len(grid[0])
        seen = set()

        for jdx in range(M):
            for idx in range(N):
                if (jdx, idx) in seen:
                    continue
                if dfs((jdx, idx), None):
                    return True

        return False

