"""
1. the first node of preorder is the root.
2. search for first node in inorder, and break into left and right subtrees.
3. this is recursive.
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:

        if not inorder:
            return None

        root_val = preorder[0]
        root_idx_inorder = inorder.index(root_val)

        root = TreeNode(root_val)

        root.left = self.buildTree(preorder[1:root_idx_inorder+1],
								   inorder[:root_idx_inorder])
        root.right = self.buildTree(preorder[root_idx_inorder+1:],
									inorder[root_idx_inorder+1:])

        return root
