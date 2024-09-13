# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import deque
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:

        traversal = []
        queue = deque()
        queue.append(root)

        while queue:

            level = []
            len_queue = len(queue)

            for idx in range(len_queue):

                node = queue.popleft()
                if node:
                    level.append(node.val)
                    queue.append(node.left)
                    queue.append(node.right)

            if len(level)>0:
                traversal.append(level)

        return traversal
