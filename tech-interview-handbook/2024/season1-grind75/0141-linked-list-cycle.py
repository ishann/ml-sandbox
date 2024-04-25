"""
Approach:
    Slow and fast pointers.

TC:
    Linear parse through the LL with one extra pass
    through the cycle but we're moving two nodes at a time.
    => O(N)

Space:
    Two pointers: slow and fast.
    => O(1)
"""

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:

        if head is None or head.next is None:
            return False

        slow = head
        fast = head.next

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow==fast:
                return True

        return False

