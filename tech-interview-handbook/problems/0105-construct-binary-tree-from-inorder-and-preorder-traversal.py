"""
Problem URL: https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal

Approach:
    First element in preorder is the root.
    Search for it in inorder and split elements into left and right subtrees.
    Recurse.
TC:
    list.index() is an O(N) op.
    Each node is built using one call so O(N) calls.
    So, O(N**2).
    Consider storing the inorder indices in a hashmap to speed things up.
Space:
    N recursive calls to buildTree - one per node being built.
    => O(N).
"""
#Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    
    def buildTree_chatGPT(self, preorder, inorder):
        """
        Discussed TC with ChatGPT. It said it was O(N) cos each node in tree built once.
        I pointed out that the .index() function is also O(N), making it O(N**2).
        It agreed, and came up with this solution to stay O(N).
        Submitting this solution results in a _significant_ TC speedup.
        """
        def buildTreeHelper(pre_start, pre_end, in_start, in_end):
            
            if pre_start > pre_end or in_start > in_end:
                return None
            
            root_val = preorder[pre_start]
            root = TreeNode(root_val)
            inorder_index = inorder_index_map[root_val]
            
            left_tree_size = inorder_index - in_start
            
            root.left = buildTreeHelper(pre_start + 1,
                                        pre_start + left_tree_size,
                                        in_start,
                                        inorder_index - 1)
            root.right = buildTreeHelper(pre_start + left_tree_size + 1,
                                        pre_end,
                                        inorder_index + 1,
                                        in_end)
            
            return root
        
        inorder_index_map = {val: idx for idx, val in enumerate(inorder)}
        return buildTreeHelper(0, len(preorder) - 1, 0, len(inorder) - 1)

    
    def buildTree(self, preorder, inorder):
        """
        Less space, less readable.
        """
        if len(preorder)==0:
            return None

        if len(preorder)==1:
            return TreeNode(val=preorder[0])
        
        idx = inorder.index(preorder[0])

        return TreeNode(val=preorder[0],
                        left=self.buildTree(preorder[1:1+len(inorder[:idx])],
                                            inorder[:idx]),
                        right=self.buildTree(preorder[1+len(inorder[:idx]):],
                                             inorder[idx+1:]))

    
    def buildTree_more_space_more_readable(self, preorder, inorder):
        """
        More space, but more readable.
        """

        if len(preorder)==0:
            return None

        if len(preorder)==1:
            return TreeNode(val=preorder[0])
        
        idx = inorder.index(preorder[0])

        # Get subtree traversals using idx.        
        left_inorder = inorder[:idx]
        right_inorder = inorder[idx+1:]

        len_left = len(inorder[:idx])
        len_right = len(inorder[idx+1:])
        
        left_preorder = preorder[1:1+len_left]
        right_preorder = preorder[1+len_left:]

        # Populate elements for returning.
        val_ = preorder[0]
        left_ = self.buildTree(left_preorder, left_inorder)
        right_ = self.buildTree(right_preorder, right_inorder)

        return TreeNode(val=val_, left=left_, right=right_)

