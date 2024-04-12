"""
Approach:
    Start with two pointers: l,r = 0, len(height)-1.
    Initialize with this container's volume.
    The smaller one of these two cannot possibly support more water, so move it inwards.
    Iterate until l!=r.

Time Complexity:
    Linear parse with two pointers.
    => O(N).

Space:
    Two int pointer variables.
    => O(1).
"""
class Solution:
    def maxArea(self, height: List[int]) -> int:
        result = -float("inf")
        l, r = 0, len(height)-1

        while l<r:
            volume = (r-l) * min(height[l], height[r])
            result = max(result, volume)
            if height[l]<height[r]:
                l+=1
            else:
                r-=1

        return result
