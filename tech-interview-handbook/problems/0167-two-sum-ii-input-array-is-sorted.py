"""
Approach:
    For every num, binary search for (target-num) in either numbers[:idx] or numbers[idx+1:].
Time Complexity:
    O(N.logN) for linear parse and binary search on each linear parse.
Space Complexity:
    O(1) as mandated by the Q itself.
"""
class Solution:
    
    def twoSum_from_solutions(self, numbers, target):
        """
        Linear parse. Not sure why I did not go to this directly...!
        TC: O(N)
        """
        l, r = 0, len(numbers)-1

        while l<r:
            total = numbers[l]+numbers[r]
            if total==target:
                break
            elif total<target:
                l+=1
            else:
                r-=1

        return [l+1, r+1]

    def twoSum(self, numbers, target):

        N = len(numbers)

        for idx in range(len(numbers)):

            l, r = idx+1, N-1
            other = target-numbers[idx]

            while l<=r:
                mid = l + (r-l)//2
                if numbers[mid]==other:
                    return [idx+1, mid+1]
                elif numbers[mid]>other:
                    r = mid-1
                else:
                    l = mid+1

