"""
Problem URL: https://leetcode.com/problems/linked-list-cycle

Approach:
    Start a slow and a fast pointer (fast moves twice as fast).
    while slow is not Fast:
        If we reach fast is None or fast.next is None, then return False.
    return True

TC:
    Linear run through LL => O(N)

Space:
    fast and slow => O(1)
"""

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head):
        
        # Empty or singleton.
        if head is None or head.next is None:
            return False

        slow = head
        fast = head.next

        while slow is not fast:
            
            if fast is None or fast.next is None:
                return False

            slow = slow.next
            fast = fast.next.next

        return True

