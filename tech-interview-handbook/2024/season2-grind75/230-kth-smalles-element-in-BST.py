"""
Inorder traversal gives sorted BST in a list.
Return the [k-1]th element (1-idx --> 0-idx).
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:

    def inorder(self, node):

        if not node:
            return []

        return self.inorder(node.left) + [node.val] + self.inorder(node.right)

    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:

        traversal = self.inorder(root)

        return traversal[k-1]
