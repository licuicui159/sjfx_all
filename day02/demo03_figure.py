# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_figure.py  窗口操作
"""
import matplotlib.pyplot as mp

mp.figure('Figure A', facecolor='gray')
mp.plot([1, 2], [3, 4])
mp.figure('Figure B', facecolor='lightgray')
mp.plot([1, 2], [4, 3])

# 重新调用Figure A，把A窗口置为当前窗口
mp.figure('Figure A')
mp.title('Figure A', fontsize=18)
mp.xlabel('Date', fontsize=14)
mp.ylabel('Price', fontsize=14)
mp.grid(linestyle=':')
mp.tight_layout()
mp.show()
