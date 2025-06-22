"""
after doing house robber 1, the insight would be whether
to include the first house or the last house.
house_robber_v2 = max(house_robber_v1(1,n-1), house_robber_v1(2,n)) 
"""
class Solution:
        
    def rob_v1(self, sub_nums: List[int]) -> int:
        
        k = len(sub_nums)

        if k==1:
            return sub_nums[0]

        g = [0]*k
        g[0], g[1] = sub_nums[0], max(sub_nums[0], sub_nums[1])

        for idx in range(2,k):
            g[idx] = max(g[idx-1], g[idx-2]+sub_nums[idx])

        return g[-1]

    def rob(self, nums: List[int]) -> int:

        n = len(nums)

        if n==1:
            return nums[0]

        if n==2:
            return max(nums[0], nums[1])

        take_first_house = self.rob_v1(nums[:n-1])
        leave_first_house = self.rob_v1(nums[1:])

        return max(take_first_house, leave_first_house)

