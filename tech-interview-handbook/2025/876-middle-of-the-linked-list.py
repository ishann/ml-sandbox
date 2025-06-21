"""
Brute Force:
1. Iterate through list and count.
2. Return node at count//2+1.

Better:
Slow-fast pointers.
"""
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:

    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:

        if head is None or head.next is None:
            return head

        slow = head
        fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        return slow

    def middleNode_v0(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        if head is None or head.next is None:
            return head

        curr = head
        count = 0
        while curr:
            count+=1
            curr = curr.next

        print(count)

        for _ in range(count//2):
            head = head.next

        return head
