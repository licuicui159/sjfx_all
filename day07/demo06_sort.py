# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo06_sort.py  排序
"""
import numpy as np

names = np.array(
    ['Mi', 'Huawei', 'Oppo', 'Vivo', 'Apple'])
prices = np.array([2999, 4999, 3999, 3999, 8888])
volumes = np.array([80, 110, 60, 70, 30])

# 排序  按价格升序排列，输出品牌列表
print(np.msort(prices))
print(names[np.argsort(prices)])
# 联合间接排序
indices = np.lexsort((-volumes, prices))
print(indices, names[indices])

# 插入排序
A = np.array([1, 3, 5, 7, 9])
B = np.array([4, 6])
indices = np.searchsorted(A, B)
print(indices)
A = np.insert(A, indices, B)
print(A)
