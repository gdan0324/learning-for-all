"""
Time:2022/11/24 10:09
Author:ECCUSYB
"""


class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        n = len(nums)
        if not nums or n < 3:
            return []
        threelist = []
        for i in range(n):
            if i > 0 and nums[i - 1] == nums[i]: continue
            for j in range(i, n):
                if j > (i+1) and nums[j - 1] == nums[j]: continue
                for k in range(j, n):
                    if k > (j + 1) and nums[k - 1] == nums[k]: continue
                    if i != j and i != k and j != k and nums[i] + nums[j] + nums[k] == 0:
                        threelist.append([nums[i], nums[j], nums[k]])
        return threelist


test = Solution()
nums = [-1, 0, 1, 2, -1, -4]
re = test.threeSum(nums)

print(re)
