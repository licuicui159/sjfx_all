# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo08_gridline.py 网格线
"""
import numpy as np
import matplotlib.pyplot as mp

y = np.array([1, 10, 100, 1000, 100, 10, 1])

mp.subplot(1, 2, 1)
# 刻度网格线
ax = mp.gca()
# 设置刻度定位器
ax.xaxis.set_major_locator(mp.MultipleLocator(1))
ax.xaxis.set_minor_locator(mp.MultipleLocator(0.1))
ax.yaxis.set_major_locator(mp.MultipleLocator(250))
ax.yaxis.set_minor_locator(mp.MultipleLocator(50))
ax.grid(which='major', axis='both',
        color='orangered', linewidth=0.5)
ax.grid(which='minor', axis='both',
        color='orangered', linewidth=0.25)
mp.plot(y)

mp.subplot(1, 2, 2)
# 刻度网格线
ax = mp.gca()
# 设置刻度定位器
ax.xaxis.set_major_locator(mp.MultipleLocator(1))
ax.xaxis.set_minor_locator(mp.MultipleLocator(0.1))
ax.yaxis.set_major_locator(mp.MultipleLocator(250))
ax.yaxis.set_minor_locator(mp.MultipleLocator(50))
ax.grid(which='major', axis='both',
        color='orangered', linewidth=0.5)
ax.grid(which='minor', axis='both',
        color='orangered', linewidth=0.25)
mp.semilogy(y)

mp.show()
