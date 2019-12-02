# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_std.py  最值
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

print(np.std(closing_prices))
print(np.std(closing_prices, ddof=1))

m = np.mean(closing_prices)
d = closing_prices - m
v = np.mean(d**2)
s = np.sqrt(v)
print(s)
