# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo04_shape.py  维度操作
"""
import numpy as np

ary = np.arange(1, 13)
print(ary, ary.shape)
# reshape()
ary2 = ary.reshape(3, 4)
print(ary2, ary2.shape)
ary[0] = 99
print(ary2, ary2.shape)
# ravel()
print(ary2.ravel())

# flatten()  复制变维
print(ary2.flatten())

# resize()
ary.shape = (2, 3, 2)
print(ary, ary.shape)
ary.resize(3, 2, 2)
print(ary, ary.shape)
