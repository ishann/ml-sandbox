"""
Create a hashmap that tracks {current graph's nodes' vals --> clone graph's nodes}
"""
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""
from typing import Optional
from collections import deque
class Solution:

    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        # This is BFS.

        if node is None:
            return None

        q = deque()
        q.append(node)

        clones = {}
        clones[node.val] = Node(node.val, [])

        while q:

            cur_node = q.popleft()
            cur_clone = clones[cur_node.val]

            for ngbr in cur_node.neighbors:

                if ngbr.val not in clones:
                    clones[ngbr.val] = Node(ngbr.val, [])
                    q.append(ngbr)

                cur_clone.neighbors.append(clones[ngbr.val])

        return clones[node.val]
