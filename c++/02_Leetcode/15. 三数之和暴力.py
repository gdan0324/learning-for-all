"""
Time:2022/11/23 22:51
Author:ECCUSYB
"""


class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        threelist = []
        for i in range(len(nums)):
            for j in range(i, len(nums)):
                for k in range(j, len(nums)):
                    if i != j and i != k and j != k and nums[i] + nums[j] + nums[k] == 0:
                        threelist.append([nums[i], nums[j], nums[k]])
        return threelist

test = Solution()
nums = [-1, 0, 1, 2, -1, -4]
re = test.threeSum(nums)

print(re)