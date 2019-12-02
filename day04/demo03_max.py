# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_max.py  最值
"""
import numpy as np
import matplotlib.pyplot as mp
import datetime as dt


def dmy2ymd(dmy):
    # 日期转换函数
    dmy = str(dmy, encoding='utf-8')
    time = dt.datetime.strptime(dmy, '%d-%m-%Y').date()
    t = time.strftime('%Y-%m-%d')
    return t

dates, opening_prices, highest_prices, \
    lowest_prices, closing_prices, volumes = \
    np.loadtxt(
        '../da_data/aapl.csv', delimiter=',',
        usecols=(1, 3, 4, 5, 6, 7),
        dtype='M8[D], f8, f8, f8, f8, f8',
        unpack=True, converters={1: dmy2ymd})

# 30天交易中的最高价格与最低价格
maxval = np.max(highest_prices)
minval = np.min(lowest_prices)
print(minval, '~', maxval)

# 获取最高价与最低价出现的日期
max_ind = np.argmax(highest_prices)
min_ind = np.argmin(lowest_prices)
print('max() date:', dates[max_ind])
print('min() date:', dates[min_ind])

# maximum  minimum
a = np.arange(1, 10).reshape(3, 3)
b = np.arange(1, 10)[::-1].reshape(3, 3)
print(a, '<- a')
print(b, '<- b')
print(np.maximum(a, b), '<- maximum()')
print(np.minimum(a, b), '<- minimum()')
