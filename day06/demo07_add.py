# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_add.py 通用函数
"""
import numpy as np

a = np.arange(1, 7)
print(a)
print(np.add(a, a))
print(np.add.reduce(a))
print(np.add.accumulate(a))
print(np.add.outer([10, 20, 30], a))   # 外和
print(np.prod(a))
print(np.cumprod(a))
print(np.outer([10, 20, 30], a))
