"""
Time:2022/11/24 10:31
Author:ECCUSYB
"""


class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        map = {}

        n = len(nums)
        if not nums or n < 3:
            return []

        for i in range(len(nums)):
            map[nums[i]] = i
        threelist = []
        # print(map) {-4: 0, -1: 2, 0: 3, 1: 4, 2: 5}
        for i in range(0, len(nums) - 2):
            x = nums[i]
            if x > 0: break
            if i > 0 and nums[i - 1] == nums[i]: continue
            for j in range(i + 1, len(nums) - 1):
                y = nums[j]
                if x + y > 0: break
                if j > i + 1 and nums[j - 1] == nums[j]: continue
                z = 0 - x - y
                if z in map.keys() and map.get(z) > j:
                    threelist.append([x, y, z])
        return threelist


test = Solution()
nums = [-1, 0, 1, 2, -1, -4]
re = test.threeSum(nums)

print(re)
