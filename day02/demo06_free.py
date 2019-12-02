# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
free.py
"""
import matplotlib.pyplot as mp

mp.figure('Flow Layout', facecolor='lightgray')
# 设置图标的位置，给出左下角点坐标与宽高即可
# left_bottom_x: 坐下角点x坐标
# left_bottom_x: 坐下角点y坐标
# width:		 宽度
# height:		 高度
# mp.axes([left_bottom_x, left_bottom_y, width, height])
mp.axes([0.03, 0.03, 0.94, 0.47])
mp.text(0.5, 0.5, '1', ha='center', va='center',
        size=36)
mp.show()
