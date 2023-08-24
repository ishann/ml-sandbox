"""
Problem URL: https://leetcode.com/problems/delete-the-middle-node-of-a-linked-list/

Approach:
    3 pointers = pre_slow, slow, fast.
        pre_slow = Parent of slow, to be able to delete slow.
        slow = TBD.
        fast = Gets us to the end of LL.
    When fast.next and fast.next.next break the while loop, we set pre_slow.next to slow.next.

TC:
    Linear parse with O(1) ops => O(N).

Space:
    3 ListNode objects => O(1).

WOF:
    head may be singleton LL.
    Check for both fast.next and fast.next.next.
"""
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteMiddle(self, head):
        
        # If only one node, delete the node and return empty LL.
        if head.next is None:
            return None

        pre_slow, slow, fast = head, head.next, head.next

        while fast.next and fast.next.next:

            pre_slow = pre_slow.next
            slow = slow.next
            fast = fast.next.next

        pre_slow.next = slow.next

        return head
        