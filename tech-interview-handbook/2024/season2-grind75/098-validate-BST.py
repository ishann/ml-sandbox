# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:

    def isValid(self, node, lower_bound, upper_bound):

        if not node:
            return True

        if not(node.val>lower_bound and node.val<upper_bound):
            return False

        isLeftValid = self.isValid(node.left, lower_bound, node.val)
        isRightValid = self.isValid(node.right, node.val, upper_bound)

        return isLeftValid and isRightValid

    def isValidBST(self, root: Optional[TreeNode]) -> bool:

        return self.isValid(root, -float("inf"), float("inf"))
