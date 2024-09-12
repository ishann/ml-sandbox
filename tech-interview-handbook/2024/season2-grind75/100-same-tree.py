"""
check if p.val==q.val
check if isSameTree(p.left, q.left)
check if isSameTree(p.right, q.right)
recursive, so take care of base case.
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:

        if p is None and q is None:
            return True

        if p is None or q is None:
            return False

        checkval = p.val==q.val
        checkleft = self.isSameTree(p.left, q.left)
        checkright = self.isSameTree(p.right, q.right)

        return True if (checkval and checkleft and checkright) else False
