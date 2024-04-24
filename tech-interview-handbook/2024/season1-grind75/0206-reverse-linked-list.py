"""
Approach:
    Use a temporary variable to exchange pointers.
    Then, advance curr to curr.next.

TC:
    Linear pass of LL with constant ops => O(N).

Space:
    A few temporary variables => O(1).
"""
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:

    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:

        curr = head
        prev = None

        while curr:

            # Use a tmp pointer to exchange.
            tmp = curr.next
            curr.next = prev
            prev = curr

            # Move curr forward.
            curr = tmp

        return prev


    def reverseList_(self, head: Optional[ListNode]) -> Optional[ListNode]:

        prev = None
        curr = head

        while curr:

            # Exchange curr and prev using tmp.
            tmp = curr.next
            curr.next = prev
            prev = curr

            # Move forward.
            curr = tmp

        return prev


