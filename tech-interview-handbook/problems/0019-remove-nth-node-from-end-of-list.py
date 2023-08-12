"""
Approach:
    2 pointers: slow and fast.
    Move fast n times before working with slow.
    Now, move both together.
    Exit such that slow is pointing to the node to be deleted's parent.
    Delete: slow.next=slow.next.next
TC:
    Linear parse with O(1) ops => O(N)
Space:
    2 ListNode objects (fast and slow) => O(1)
WOF:
    fast may not exist after n steps => delete head.
"""
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head, n):
        
        fast = head
        slow = head

        for _ in range(n):
            fast = fast.next

        if fast is None:
            return head.next

        while fast.next:
            fast = fast.next
            slow = slow.next

        slow.next = slow.next.next

        return head

