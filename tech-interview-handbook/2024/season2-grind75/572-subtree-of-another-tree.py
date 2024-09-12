"""
Fortunately, solved isSameTree right before this.
Breaks down into an extension of isSameTree.
Be careful with recursing using isSameTree while also recursing with isSubtree.
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:

    def isSameTree(self, p, q):

        if p is None and q is None:
            return True

        if p is None or q is None:
            return False

        return True if (p.val==q.val and
                        self.isSameTree(p.left, q.left) and
                        self.isSameTree(p.right, q.right)) else False

    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:

        # If both None, then None is indeed subtree of None.
        if root is None and subRoot is None:
            return True

        # root is None, but subRoot is not None.
        if root is None:
            return False

        return True if (self.isSameTree(root, subRoot) or
                        self.isSubtree(root.left, subRoot) or
                        self.isSubtree(root.right, subRoot)) else False

