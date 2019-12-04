# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo06_numpy.py  通用函数
"""
import numpy as np
# 裁剪
ary = np.arange(1, 11)
print(ary)
print(ary.clip(min=3, max=7))

# 压缩
print(ary)
print(ary.compress(
    np.all([ary % 2 == 0, ary % 3 == 0], axis=0)))
