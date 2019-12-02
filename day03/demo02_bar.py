# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
mean.py  柱状图
"""
import numpy as np
import matplotlib.pyplot as mp

apples = np.array([
    87, 23, 45, 68, 23, 52, 38, 45, 62, 39, 23, 42])
oranges = np.array([
    47, 56, 37, 84, 59, 23, 84, 56, 29, 43, 85, 35])

mp.figure('Bar Chart', facecolor='lightgray')
mp.title('Bar Chart', fontsize=18)
mp.grid(linestyle=':', axis='y')
mp.xlabel('Date', fontsize=14)
mp.ylabel('Volume', fontsize=14)
x = np.arange(apples.size)
mp.bar(x - 0.2, apples, 0.4, label='Apples',
       color='dodgerblue')
mp.bar(x + 0.2, oranges, 0.4, label='Oranges',
       color='orangered')

# 优化x轴刻度文本
mp.xticks(x, [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

mp.legend()
mp.show()
