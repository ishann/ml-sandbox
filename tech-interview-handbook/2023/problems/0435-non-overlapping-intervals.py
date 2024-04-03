"""
Problem URL: https://leetcode.com/problems/non-overlapping-intervals

Method:
    Greedy. Sort by either start or end. Then compare and pick by end or start.
    The idea is that if we start choosing in order of earliest end time,
    we are (greedily) leaving the most amount of timeperiods free for the remaining intervals.
Time Complexity:
    Python's sorted takes O(N.logN).
Space Complexity:
    O(1).
"""
class Solution:

    def eraseOverlapIntervals(self, intervals):

        intervals.sort(key=lambda x: x[1])

        end = -50001
        counter_ = 0

        for [b,e] in intervals:

            if b>=end:
                end=e
            else:
                counter_+=1
        
        return counter_