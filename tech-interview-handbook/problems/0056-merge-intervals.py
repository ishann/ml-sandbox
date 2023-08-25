"""
Problem URL: https://leetcode.com/problems/merge-intervals

Approach:
    Sort on start times.
    Initialize answer with the first interval.
    Insert each subsequent interval while either merging or just adding.

Time Complexity:
    One pass through the array => O(N).

Space Complexity:
    Excluding return space, none => O(1).

WOF:
    [x,x] type intervals.
"""
class Solution:
    def merge(self, intervals):

        if len(intervals)==1:
            return intervals

        intervals.sort(key=lambda x: x[0])

        idx, len_ = 1, len(intervals)
        ans = [intervals[0]]

        while idx<len_:
            if intervals[idx][0]>ans[-1][1]:
                ans.append(intervals[idx])
            else:
                ans[-1][1] = max(ans[-1][1], intervals[idx][1])
            idx+=1

        return ans



