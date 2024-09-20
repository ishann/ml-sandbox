"""
1. Do a preorder DFS traversal. Add a special char for None.
   Lists make things easier. ",".join(list) at the end.
2. Run another DFS while tracking the idx.
"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """

        traversal = []

        def dfs(node):

            if node is None:
                traversal.append("Null")
                return

            traversal.append(str(node.val))

            dfs(node.left)
            dfs(node.right)

        dfs(root)

        return ",".join(traversal)


    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """

        traversal = data.split(",")
        idx = 0

        def dfs():
            nonlocal idx

            if traversal[idx]=="Null":
                idx += 1
                return

            node = TreeNode(int(traversal[idx]))
            idx += 1

            node.left = dfs()
            node.right = dfs()

            return node

        return dfs()

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))
