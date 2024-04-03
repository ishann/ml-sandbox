"""
Problem URL: https://leetcode.com/problems/clone-graph

Approach:
    Start from node.
    Do a BFS, keeping track of visited nodes.
    While doing the iterative BFS, create clones for each new node we encounter.
    Make sure to populate the adjacency lists too.

TC:
    Iterative BFS with O(1) ops => O(V+E)

Space:
    O(V) for clones.
    O(V) for queue.
"""
from collections import deque

# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

class Solution:

    def cloneGraph(self, node):

        if not node:
            return node

        queue = deque([node])
        clones = {node.val: Node(node.val, [])}

        # Do BFS
        while queue:
            # Get a node from the queue.
            cur_node = queue.popleft()
            # Get its corresponding clone.
            cur_clone = clones[cur_node.val]
            
            # Iterate over neighbors of cur_node.
            for ngbr in cur_node.neighbors:
                # If ngbr.val not in clones, add it.
                # Also, put ngbr in queue.
                if ngbr.val not in clones:
                    clones[ngbr.val] = Node(ngbr.val, [])
                    queue.append(ngbr)
                # Add the cloned node to cur_clone neighbors.
                cur_clone.neighbors.append(clones[ngbr.val])
        
        return clones[node.val]

