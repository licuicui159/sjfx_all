# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo09_bitwise.py
"""
import numpy as np

a = np.array([0, -1, 2, -3, 4, -5])
b = np.array([0, 1, 2, 3, 4, 5])
print(a, b)
c = a ^ b
# c = a.__xor__(b)
# c = np.bitwise_xor(a, b)
print(c)
# 找到c数组中哪一个位置是负数
print(np.where(c < 0)[0])


d = np.arange(1, 21)
print(d)
e = d & (d - 1)
e = d.__and__(d - 1)
e = np.bitwise_and(d, d - 1)
print(e)

a = np.arange(1000000)
print(a[a & (a - 1) == 0])
