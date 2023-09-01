"""
Approach:
    Init odd and even ListNodes.
    While (odd.next and even.next):
        Skip and connect for both
    Connect odd tail to even head.
TC:
    Linear parse with O(1) ops => O(N).
Space:
    odd, even, even_head => O(1).
"""
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head):
        
        # If empty or singleton.
        if not head or not head.next:
            return head

        odd = head
        even = head.next

        # To link odd tail to even head.
        even_head = even

        # While both nexts exist.
        while odd.next and even.next:

            odd.next = odd.next.next
            even.next = even.next.next

            odd = odd.next
            even = even.next

        # Link odd and even.
        odd.next = even_head

        return head

