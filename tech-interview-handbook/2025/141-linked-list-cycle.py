"""
Use slow and fast pointers.
Make sure to terminate when slow.next is None or maybe even if fast.next is None.
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
        fast = head

        while slow and slow.next is not None and fast and fast.next is not None:
            slow=slow.next
            fast=fast.next.next
            if slow==fast:
                return True

        return False
