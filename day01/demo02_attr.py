# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo02_attr.py  属性基本操作
"""
import numpy as np

ary = np.arange(1, 9)
print(ary)
# shape： 维度
print(ary.shape)
ary.shape = (2, 4)
print(ary, ary.shape)

# dtype:  元素类型
print(ary, ary.dtype)
# ary.dtype = 'float32'
# print(ary, ary.dtype)
ary = ary.astype('float32')
print(ary, ary.dtype)

# size:  元素个数
print(ary, ' size:', ary.size, ' len():', len(ary))

# 索引访问
print(ary[0], ' <- ary[0]')
print(ary[0][1], ' <- ary[0][1]')
print(ary[0, 1], ' <- ary[0, 1]')
# 迭代
ary = np.arange(1, 28)
ary.shape = (3, 3, 3)
print(ary, ary[1, 1, 1])
# 遍历三维数组
for i in range(ary.shape[0]):
    for j in range(ary.shape[1]):
        for k in range(ary.shape[2]):
            print(ary[i, j, k], end=' ')
