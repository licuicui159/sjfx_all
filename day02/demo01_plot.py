# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_plot.py 基本绘图
"""
import numpy as np
import matplotlib.pyplot as mp

xarray = np.array([1, 2, 3, 4, 5, 6, 7, 8])
yarray = np.array([6, 17, 3, 41, 36, 1, 53, 12])
mp.plot(xarray, yarray)
# 绘制水平线
mp.hlines(30, 2, 7)
# 绘制一堆垂直线
mp.vlines([2, 3, 4, 5],
          [10, 20, 30, 40], [40, 50, 60, 70])
mp.show()
