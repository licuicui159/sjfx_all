# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_stack.py
"""
import numpy as np
a = np.arange(1, 7).reshape(2, 3)
b = np.arange(7, 13).reshape(2, 3)
print(a)
print(b)
# vstack()  vsplit()
c = np.vstack((a, b))
print(c)
a, b = np.vsplit(c, 2)
print(a)
print(b)

# hstack()  hsplit()
c = np.hstack((a, b))
print(c)
a, b = np.hsplit(c, 2)
print(a)
print(b)

# dstack()  dsplit()
c = np.dstack((a, b))
print(c)
a, b = np.dsplit(c, 2)
print(a)
print(b)

# 简单一维数组的组合方案
a = np.arange(1, 9)
b = np.arange(11, 19)
c = np.row_stack((a, b))
print(c)
print(c[:, :3])
c = np.column_stack((a, b))
print(c)
print(c[:, 0])
print(c[:, :1])
