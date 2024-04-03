"""
Problem URL: https://leetcode.com/problems/invert-binary-tree

Approach:
    Recursion.
    My code is slightly less efficient but much more verbose.
TC:
    O(N) because we visit each node once and perform O(1) ops at each node.
Space:
    O(N) on fncn call stack for worst-case, else O(logN).
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def invertTree(self, root):
        
        if root is None:
            return None

        if root.left is not None:
            left_subtree = self.invertTree(root.left)
        else:
            left_subtree = None

        if root.right is not None:
            right_subtree = self.invertTree(root.right)
        else:
            right_subtree = None

        return TreeNode(val=root.val,
                        left=right_subtree,
                        right=left_subtree)

