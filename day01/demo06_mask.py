# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo06_mask.py 掩码
"""
import numpy as np

ary = np.arange(1, 10)
print(ary)
mask = ary % 2 == 0
print(mask)
print(ary[mask])
# 掩到相应元素之后，为当前位置的元素赋值
ary[mask] = 100
print(ary)
# 输出100以内3与7的倍数
a = np.arange(100)
print(a[(a % 3 == 0) & (a % 7 == 0)])

# 索引掩码
a = np.array([81, 72, 23, 56, 12, 87, 54, 12, 83, 45])
mask = [3, 6, 2, 1, 3, 6, 2, 1, 3, 6, 2, 1, 3, 6, 2,
        1, 3, 6, 2, 1, 3, 6, 2, 1, 3, 6, 2, 1, 3, 6, 2, 1]
print(a[mask])

names = np.array(['Mi', 'Oppo', 'Vivo', 'Huawei'])
prices = np.array([3999, 2999, 4999, 18999])
i = np.argsort(prices)
print(names[i])
