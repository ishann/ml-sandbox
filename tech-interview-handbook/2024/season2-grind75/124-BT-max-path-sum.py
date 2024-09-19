"""
Requires DFS.
Requires a non-local variable.

Two possibilities:
1. either split at a sub-trees root (which means answer is f(node.left)+node.val+f(node.right))
2. propagate upwards from sub-trees root node.val + max(f(node.left), f(node.right))

make sure that we account for node/ entire subtree giving -ve gain.
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:

        res = -float("inf")

        def find_max_gain(node):

            nonlocal res

            if node is None:
                return 0

            max_left_gain = max(find_max_gain(node.left), 0)
            max_right_gain = max(find_max_gain(node.right), 0)

            node_max_gain = max_left_gain + node.val + max_right_gain

            res = max(res, node_max_gain)

            return node.val + max(max_left_gain, max_right_gain)

        find_max_gain(root)

        return res
