class Solution:
    def singleNumber(self, nums):
        for i in nums:
            dict={}
            dict.setdefault(nums[i])

s=Solution()

print(s.singleNumber([2,2,1]))