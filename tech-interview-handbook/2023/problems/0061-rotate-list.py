"""
Approach: 
    Get length of LL. Standardize k according to length.
    Add head to next of last element to get circular list.
    Iterate through the circular LL and cut at length-k-1.
TC:
    Linear pass with O(1) ops => O(N)
Space:
    O(1)
"""
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def rotateRight(self, head, k):
        
        if not head:
            return None

        last = head

        length = 1
        while last.next:
            last = last.next
            length+=1
        
        k %= length
        last.next = head

        temp = head
        for _ in range(length-k-1):
            temp = temp.next

        rotated = temp.next
        temp.next = None

        return rotated