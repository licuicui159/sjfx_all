# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_random.py 随机数
"""
import numpy as np

# 命中率0.3  投10次  进几个
r = np.random.binomial(10, 0.3, 100000)
for i in range(11):
    print(i, ':', (r == i).sum() / 100000)


r = np.random.binomial(3, 0.6, 100000)
for i in range(4):
    print(i, ':', (r == i).sum() / 100000)

# 超几何分布
print('-' * 45)
r = np.random.hypergeometric(7, 3, 3, 100000)
for i in range(4):
    print(i, ':', (r == i).sum() / 100000)
