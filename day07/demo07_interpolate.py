# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_interpolate.py 插值器
"""
import numpy as np
import matplotlib.pyplot as mp

min_val = -50
max_val = 50

x = np.linspace(min_val, max_val, 15)
y = np.sinc(x)


mp.grid(linestyle=':')
mp.scatter(x, y, color='dodgerblue', s=80,
           label='Samples')
# 线性插值器
import scipy.interpolate as si
# 返回的linear为线性插值器函数
linear = si.interp1d(x, y)
px = np.linspace(min_val, max_val, 1000)
py = linear(px)
mp.plot(px, py)

# 返回的cubic为cubic插值器函数
cubic = si.interp1d(x, y, kind='cubic')
px = np.linspace(min_val, max_val, 1000)
py = cubic(px)
mp.plot(px, py)


mp.legend()
mp.show()
