"""
 Approach:
     At each node, make a decision to rob it or not.
     If we rob it, we cannot rob either of its children.
     If we do not rob it, we can rob both children.
TC:
    DFS through the BT with O(1) ops => O(N)
Space:
    A few ints in the DFS routine => O(N)
    Recursion stack => O(N)
    Hence, O(N)
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:

    def rob(self, root):

        def custom_dfs(node):
            """
            Basically, does everything.
            """
            if not node:
                return [0, 0]

            amount_left_child = custom_dfs(node.left)
            amount_right_child = custom_dfs(node.right)

            rob = node.val + amount_left_child[-1] + amount_right_child[-1]
            not_rob = max(amount_left_child) + max(amount_right_child)

            return [rob, not_rob]

        return max(custom_dfs(root))
