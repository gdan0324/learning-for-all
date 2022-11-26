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
        n = len(nums)
        if not nums or n < 3:
            return []
        nums.sort()
        res = []
        for i in range(n):
            # 如果第一个值大于零，直接返回空
            if nums[i] > 0:
                return res
            # 如果当前值等于上一个值，跳过，进入下一次循环，去除重复值
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            L = i + 1
            R = n - 1
            while (L < R):  # 如果 L>R 或者 L=R 就结束
                if nums[i] + nums[L] + nums[R] == 0:
                    res.append([nums[i], nums[L], nums[R]])
                    while L < R and nums[L] == nums[L + 1]:
                        L = L + 1
                    while L < R and nums[R] == nums[R - 1]:
                        R = R - 1
                    L = L + 1
                    R = R - 1
                # 如果三数之和大于零，就将R--
                elif nums[i] + nums[L] + nums[R] > 0:
                    R = R - 1
                else:
                    L = L + 1
        return res


test = Solution()
nums = [-1, 0, 1, 2, -1, -4]
re = test.threeSum(nums)

# print(re)

matrix03 = [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]]
print("matrix03:", matrix03)

matrix04 = {{0, 0, 0},
            {0, 0, 0},
            {0, 0, 0}}

print(matrix04)
