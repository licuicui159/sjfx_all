# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo06_apply.py   轴向汇总
"""
import numpy as np

data = np.array([[80, 81, 82],
                 [92, 82, 88],
                 [94, 83, 87],
                 [98, 84, 84]])
# 统计每个人三门成绩的平均分：
for row in range(len(data)):
    print(data[row].mean())
# 统计每一门成绩的最高分：
for col in range(data.shape[1]):
    print(np.max(data[:, col]))


def func(data):
    return np.max(data), np.min(data), np.mean(data)

# numpy提供的轴向汇总相关API：
r = np.apply_along_axis(func, 1, data)
print(r)
