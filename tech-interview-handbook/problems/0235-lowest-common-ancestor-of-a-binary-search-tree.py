"""
Problem URL: https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree

Method 1: Binary Tree (does not exploit BST structure).
    Approach:
        Create a hashmap for {node:parent} pairs.    
        
        Get path, say p_path, from p to root using hashmap.
        
        Go from q to root until we encounter an element in p_path.
        Return this element.
Method 2: Binary Search Tree.
    Approach:
        If both p and q on left subtree, descend left.
        Elif both p and q on right subtree, descend right.
        Else, we are at LCA. Return it.

"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    
    def lowestCommonAncestor_method2(self, root, p, q):
    
        p_val, q_val = p.val, q.val
        
        while root:
            
            par_val = root.val
            
            if p_val<par_val and q_val<par_val:
                root = root.left
            elif p_val>par_val and q_val>par_val:
                root = root.right
            else:
                return root
            

    def lowestCommonAncestor_method1(self, root, p, q):
        
        if root==p or root==q:
            return root

        # Create a hashmap of parents.
        parent = {root: None}
        stack = [root]
        while p not in parent or q not in parent:
            node = stack.pop()
            if node.left:
                parent[node.left] = node
                stack.append(node.left)
            if node.right:
                parent[node.right] = node
                stack.append(node.right)

        # Get path of p from node to root.
        p_path = []
        while p:
            p_path.append(p)
            p = parent[p]

        # Search for q while traversing from node to root in path of p.
        while q not in p_path:
            q = parent[q]
        
        return q