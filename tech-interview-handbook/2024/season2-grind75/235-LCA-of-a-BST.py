"""
Insight: LCA will have p on one side and q on the other side.
Since we're recursing, we need to be careful about base cases.
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

        """
        p and q will be on opposite sides of LCA (or one of p or q will be LCA itself).
        """

        node = root

        while node:

            if node.val>p.val and node.val>q.val:
                node = node.left
            elif node.val<p.val and node.val<q.val:
                node = node.right
            else:
                return node
