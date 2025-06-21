"""
Iterate and populate until both lists exist.
Outside, append entirely if either list remains.
"""
class Solution:

    def mergeTwoLists_v0(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        
        if list1 is None:
            return list2
        if list2 is None:
            return list1

        list_ = ListNode()
        dummy = list_

        while list1 and list2:
            list_.next = ListNode()
            list_ = list_.next
            if list1.val < list2.val:
                list_.val = list1.val
                list1 = list1.next
            else:
                list_.val = list2.val
                list2 = list2.next            

        if list1:
            list_.next = list1
        if list2:
            list_.next = list2

        return dummy.next
