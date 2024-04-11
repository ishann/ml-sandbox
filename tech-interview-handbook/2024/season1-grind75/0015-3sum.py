"""
Method 2:
    Approach:
        Sort the array.
        Always move pointer forward until a different num1 is encountered.
        While solving TwoSum in sorted array, also move pointer forward
        until a different num2/ num3 is encountered.

    Time Complexity:
        Sorting takes O(N logN).
        Linear parse for num1 takes O(N).
            Sorted TwoSum takes O(N) with two pointers.
        => O(N logN + N^2)
        => O(N^2).

    Space:
        Hashmap will take O(N).
        Assuming sorting takes O(N) as well...
        => O(N).

Method 1 (has duplicate triplets):
    Approach:
        Linear parse per element.
        Per iteration, apply the familiar TwoSum pattern where target = -element.

    Time Complexity:
        Linear parse => O(N).
        TwoSum with HashMap => O(N).
        => O(N^2).

    Space:
        Hashmap will store at most N elements => O(N).
"""
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:

        result = []
        nums.sort()

        for idx, num1 in enumerate(nums):

            if idx>0 and num1==nums[idx-1]:
                continue

            l, r = idx+1, len(nums)-1
            while l<r:
                num2 = nums[l]
                num3 = nums[r]

                if (num1+num2+num3)>0:
                    r-=1
                elif (num1+num2+num3)<0:
                    l+=1
                else:
                    result.append([num1, num2, num3])
                    l+=1
                    while nums[l]==nums[l-1] and l<r:
                        l+=1

        return result


    def threeSum_does_not_account_for_duplicates(self, nums: List[int]) -> List[List[int]]:

        result = []

        for idx, numi in enumerate(nums):

            target = -numi

            hashmap = {}

            for jdx, numj in enumerate(nums[idx+1:]):
                hashmap[numj] = jdx+idx+1

            for kdx, numk in enumerate(nums[idx+1:]):
                if target-numk in hashmap:
                    if kdx+idx+1 != hashmap[target-numk]:
                        num1 = nums[idx]
                        num2 = nums[kdx+idx+1]
                        num3 = nums[hashmap[target-numk]]
                        resulti = [num1, num2, num3]
                        print(resulti)
                        result.append(resulti)

        return result


