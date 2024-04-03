"""
Problem URL: https://leetcode.com/problems/insert-interval

3 phases:
1. while interval[i][1]<newInterval[0]: append intervals[i] to result.
2. merge until newInterval[1]>=intervals[i][0]. insert newInterval into result.
3. append remaining intervals[i] to result.

TC:
    Linear pass over intervals.
    => O(N)

Space:
    O(1)
"""
class Solution:
    def insert(self, intervals, newInterval):

        if not intervals:
            return [newInterval]

        N = len(intervals)

        result = []
        i = 0
        
        # Before overlap = just insert intervals[i].
        while i<N and intervals[i][1]<newInterval[0]:
            result.append(intervals[i])
            i+=1

        # When overlap = merge and insert newInterval.
        while i<N and newInterval[1]>=intervals[i][0]:
            newInterval[0] = min(intervals[i][0], newInterval[0])
            newInterval[1] = max(intervals[i][1], newInterval[1])
            i+=1
        result.append(newInterval)
        
        # After overlap = just insert intervals[i].
        while i<N:
            result.append(intervals[i])
            i+=1

        return result

