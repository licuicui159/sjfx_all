# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo02_piecewise.py    数组处理函数
"""
import numpy as np

ary = np.array([87, 12, 39, 23, 54, 12, 83, 45, 12, 98])
# 输出及格不及格
r = np.piecewise(ary, [ary < 60, ary > 60], [0, 1])
print(r, r.sum())
