"""
Problem URL: 

Approach:
    Greedy. Sort by end of each interval.
    Shoot an arrow through the end of the first sorted interval and record "last_shot".
    Shoot a new arrow everytime a new interval begins after "last_shot".
    Remember, shoot any new arrow through the end of an interval.
TC:
    Sorting takes O(NlogN)
    Linear parse is O(N).
    Thus, limiting factor is the sort in the beginning => O(NlogN).
Space:
    Two ints (arrows and last_shot) => O(1)
    (Arrows is actually in the return space.)
"""
class Solution:
    def findMinArrowShots(self, points):
        
        if len(points)==1:
            return 1

        arrows = 0

        points.sort(key=lambda x: x[1])

        # Shoot the first arrow.
        last_shot = points[0][1]
        arrows+=1

        # Iterate and figure out how many more arrows needed.
        for point in points[1:]:
            
            [beg, end] = point

            if last_shot<beg:
                arrows+=1
                last_shot = end

        return arrows
            
