"""
Problem URL: https://leetcode.com/problems/path-sum-ii

Approach:
    Do all possible DFS paths with the following terminating conditions:
        If running path sum exceeds target, return.
        If running path sum equals target and current node is a lead, add path to result.
TC:
    Each node visited once in DFS => O(N).
    But, the res.append() and path+[node.val] are also O(N).
    => O(N**2)
Space:
    Depth of root, which in worst case is O(N) (we don't know if BT is balanced).
    Thus, stack for the recursive calls will take O(N).
    => O(N)
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
        
    def pathSum(self, root, targetSum):

        res = []

        def dfs(node, sum_, path, res):

            if node:
                if not node.left and not node.right and node.val==sum_:
                    res.append(path+[node.val])
                dfs(node.left, sum_-node.val, path+[node.val], res)
                dfs(node.right, sum_-node.val, path+[node.val], res)

        dfs(root, targetSum, [], res)

        return res

