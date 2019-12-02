# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_ndarray.py
"""
import numpy as np
# 1.
ary = np.array([1, 2, 3, 4, 5, 6])
print(ary, type(ary), ary[0])
print(ary + 2)
print(ary * 3)
print(ary > 3)
print(ary + ary)
# 2.  arange()
ary = np.arange(10)
print(ary)
print(np.arange(1, 5))
# 3.  zeros()  ones()
ary = np.zeros(6, dtype='int32')
print(ary, '<- zeros()')
ary = np.ones((2, 3), dtype='float32')
print(ary, '<- ones()')

# 4. ones_like()  zeros_like()
ary = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(ary)
ol = np.ones_like(ary)
print(ol, '<- ones_like()')
