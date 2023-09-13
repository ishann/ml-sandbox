"""
Problem URL: https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree

Approach:
    Create a hashmap for {node:parent} pairs.    
    
    Get path, say p_path, from p to root using hashmap.
    
    Go from q to root until we encounter an element in p_path.
    Return this element.

TC:
    O(N) to create hashmap.
    O(logN) to create p_path.
    O(logN) to search for q in p_path.
    => O(N)

Space:
    Hashmap takes O(N).
    p_path takes O(logN).
    => O(N)
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def lowestCommonAncestor(self, root, p, q):
        
        if root==p or root==q:
            return root

        # Create a hashmap to point to parents.
        stack = [root]
        parent = {root: None}
        
        # Looked at solutions and we don't need to go through the entire BST.
        # We could stop when both p and q become keys in parent:
        # while p not in parent or q not in parent.
        while stack:
            node = stack.pop()
            if node.left:
                parent[node.left] = node
                stack.append(node.left)
            if node.right:
                parent[node.right] = node
                stack.append(node.right)

        # Get path from p to root.
        p_path = []
        while p:
            p_path.append(p)
            p = parent[p]

        # Look for q in p_path.
        while q not in p_path:
            q = parent[q]
        
        return q

