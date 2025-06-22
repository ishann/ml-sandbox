"""
Iterate through intervals until an end_i>start_new.
New interval's start is min(start_i, start_new).
Continue merging until an start_j>end_new.
Add in the remaining intervals if any.
"""
class Solution:
    def insert(self, intervals: List[List[int]], new_interval: List[int]) -> List[List[int]]:
        
        result = []
        start_new, end_new = new_interval

        idx = 0
        n = len(intervals)

        # Add interval[idx] to result until end[idx]<start_new.
        while idx<n and intervals[idx][1]<start_new:
            result.append(intervals[idx])
            idx+=1

        # Merge. If non-overlapping loop will never begin and consume the edge case.
        while idx<n and intervals[idx][0]<=end_new:
            start_new = min(start_new, intervals[idx][0])
            end_new = max(end_new, intervals[idx][1])
            idx+=1
        result.append([start_new, end_new])
        
        # Add any that remain.
        while idx<n:
            result.append(intervals[idx])
            idx+=1

        return result
