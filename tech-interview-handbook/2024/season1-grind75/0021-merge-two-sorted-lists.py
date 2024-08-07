"""
Approach:

    while list1.next and list2.next:
        # Compare and add to dummy node.
        # Advance pointers accordingly.

    # Append if either one of the two lists still remains.

TC:
    Linear parse over both lists with constant ops.
    => O(m+n)

Space:
    One dummy variable, independent of lengths of either lists.
    => O(1).

WOF:
    No copies.
    A list may remain when the other terminates the initial loop.
"""
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:

    def mergeTwoLists(self,
                      l1: Optional[ListNode],
                      l2: Optional[ListNode]) -> Optional[ListNode]:

        # What we return. Technically, we return head.next.
        head = ListNode()

        # What merges the LL.
        tail = head

        while l1 and l2:

            if l1.val<=l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next

        if l1:
            tail.next = l1
        elif l2:
            tail.next = l2

        return head.next


    def mergeTwoLists_(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:

        # head will stay at beginning, and used to return.
        # tail will keep advancing.
        head = ListNode()
        tail = head

        while l1 and l2:

            if l1.val < l2.val:
                tail.next = l1
                l1 = l1.next
            else:
                tail.next = l2
                l2 = l2.next
            tail = tail.next

        if l1:
            tail.next = l1
        elif l2:
            tail.next = l2

        return head.next


