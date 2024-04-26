"""
Approach:
    Slow-fast to mid of list.
    Reverse the 2nd half.
    Merge alternatively.

TC:
    O(N).

Space:
    O(1).

"""
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """

        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        second = slow.next
        slow.next = None
        prev = None

        while second:
            tmp = second.next
            second.next = prev
            prev = second
            second = tmp

        first, second = head, prev

        while second:
            tmp_1 = first.next
            tmp_2 = second.next
            first.next = second
            second.next = tmp_1
            first = tmp_1
            second = tmp_2


