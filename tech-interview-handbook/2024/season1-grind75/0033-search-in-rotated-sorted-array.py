"""
[4,5,6,7,0,1,2] and 0

Method 1 : NeetCode
    Approach:
        Consider whether mid is part of left subarray
        or right subarray. Consider wrap-around and
        all possible cases.
    TC:
        Binary search => O(log N)
    Space:
        l, r => O(1).

    PS: I don't like this solution, because it deviates
        from several Binary Search norms.

Method 2 : LeetCode Forums
    Approach:

    TC:

    Space:


"""
class Solution:

    def search(self, nums: List[int], target: int) -> int:

        l, r = 0, len(nums)

        while l<r:

            mid = (l+r)//2
            if nums[mid]>nums[0] and target<nums[0]:
                l=mid+1
            elif nums[mid]<nums[0] and target>=nums[0]:
                r=mid
            elif nums[mid]<target:
                l=mid+1
            elif nums[mid]>target:
                r=mid
            else:
                return mid

        return -1

    def search_top_LC_soln(self,
                           nums: List[int],
                           target: int) -> int:

        l, r = 0, len(nums)-1

        # Find pivot of rotation.
        while l<r:
            mid = (l+r)//2
            if nums[mid]>nums[r]:
                l=mid+1
            else:
                r=mid

        rot=l

        # Do binary search accounting for pivot of rotation.
        l, r=0, len(nums)-1
        while l<=r:
            mid = (l+r)//2
            real_mid = (mid+rot)%len(nums)
            if nums[real_mid]==target:
                return real_mid
            if nums[real_mid]<target:
                l=mid+1
            else:
                r=mid-1

        return -1

    def search_try1(self, nums: List[int], target: int) -> int:

        l, r = 0, len(nums)

        while l<r:
            mid=(l+r)//2
            if target<nums[0]<nums[mid]:
                l=mid+1
            elif target>=nums[0]>nums[mid]:
                r=mid
            elif nums[mid]<target:
                l=mid+1
            elif nums[mid]>target:
                r=mid
            else:
                return mid

        return -1


