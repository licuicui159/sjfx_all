# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_vec.py  函数矢量化
"""
import numpy as np
import math as m


def foo(x, y):
    return m.sqrt(x**2 + y**2)

a, b = 3, 4
print(foo(a, b))

a, b = np.array([3, 4, 5]), np.array([4, 5, 6])
# print(foo(a, b))
# 矢量化foo函数，使之可以处理数组数据
foo_vec = np.vectorize(foo)
print(foo_vec(a, b))
b = 5
print((np.vectorize(foo)(a, b)).dtype)

# 基于frompyfunc函数实现函数矢量化
foo_func = np.frompyfunc(foo, 2, 1)
print(foo_func(a, b).dtype)
