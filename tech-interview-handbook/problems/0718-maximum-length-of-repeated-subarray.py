"""
Problem URL: https://leetcode.com/problems/maximum-length-of-repeated-subarray

Approach:
    M, N = len(nums1), len(nums2)
    0s => DP over len(nums1)xlen(nums2).
    max_len = 0

    loop: idx from [1, len(nums1)]:
        loop: jdx from [1, len(nums2)]:
            if nums1[idx-1]==nums2[jdx-1]:
                dp[idx][jdx] = dp[idx-1][jdx-1]+1
            else:
                dp[idx][jdx] = 0
            max_len = max(max_len, dp[idx][jdx])

    return max_len

TC:
    O(MxN)
SC:
    O(MxN)
"""
class Solution:
    def findLength(self, nums1, nums2):

        M, N = len(nums1), len(nums2)

        dp = [[0]*(N+1) for _ in range(M+1)]

        max_len = 0

        for idx in range(1, M+1):
            for jdx in range(1, N+1):
                if nums1[idx-1]==nums2[jdx-1]:
                    dp[idx][jdx] = dp[idx-1][jdx-1]+1
                else:
                    dp[idx][jdx] = 0
                max_len = max(max_len, dp[idx][jdx])

        return max_len
