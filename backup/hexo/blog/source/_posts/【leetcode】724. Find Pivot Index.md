---
title: 【leetcode】724. Find Pivot Index
date: 2018-1-18 11:21:55
thumbnail: https://raw.githubusercontent.com/xiongzongyang/hexo_photo/master/c.jpg
tags: leetcode
categories: [C/C++,leetcode]
---
> Given an array of integers nums, write a method that returns the "pivot" index of this array.
We define the pivot index as the index where the sum of the numbers to the left of the index is equal to the sum of the numbers to the right of the index.
If no such index exists, we should return -1. If there are multiple pivot indexes, you should return the left-most pivot index.
<!--more-->

### 1 基本思路

可以利用空间换时间，依次将从左边累加和从右边累加的结果分别保存在两个数组中，然后逐一比较这两个累加数组，当相等，则返回当前下标。
为什么两个累加数组中，相同索引i所对应的数组元素值想等，该下i标就是所要求的呢？因为左边累加的数组减去当前元素，就是i左边所有元素之和，而右累加数组减去当前元素，就是i右边元素之和，而减去的元素值是同一个，所有i满足条件。
### 2 完整代码

```
class Solution {
public:
    int pivotIndex(vector<int>& nums) {
        //获取数组中元素个数
        int len=nums.size();
        int* a=new int[len];
        int* b=new int[len];
        if(len==0)
            return -1;
        if(len==1)
            return 0;
        int sumi=0;
        int sumj=0;
        int i=0;
        int j=len-1;
        for(i=0;i<len;i++)
        {
            sumi=nums[i]+sumi;
            a[i]=sumi;
        }
        for(j=len-1;j>-1;j--)
        {
            sumj=sumj+nums[j];
            b[j]=sumj;
        }
        for(i=0;i<len;i++)
        {
            if(a[i]==b[i])
                return i;
        }
        return -1;
    }
};
```
