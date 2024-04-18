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

Method 2 : LeetCode Forums
    Approach:

    TC:

    Space:


"""
class Solution:

    def search(self, nums: List[int], target: int) -> int:

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

    def search_neetcode(self, nums: List[int], target: int) -> int:

        l,r=0,len(nums)-1

        while l<=r:
            mid=(l+r)//2
            if target==nums[mid]:
                return mid

            if nums[mid]>=nums[l]:
                if target<nums[l] or target>nums[mid]:
                    l=mid+1
                else:
                    r=mid-1
            else:
                if target<nums[mid] or target>nums[r]:
                    r=mid-1
                else:
                    l=mid+1

        return -1
