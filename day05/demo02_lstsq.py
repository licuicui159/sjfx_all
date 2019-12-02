# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo02_lstsq.py  线性拟合绘制趋势线
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
    lowest_prices, closing_prices = \
    np.loadtxt(
        '../da_data/aapl.csv', delimiter=',',
        usecols=(1, 3, 4, 5, 6),
        dtype='M8[D], f8, f8, f8, f8',
        unpack=True, converters={1: dmy2ymd})

# 画图
mp.figure('AAPL', facecolor='lightgray')
mp.title('AAPL', fontsize=18)
mp.grid(linestyle=':')
mp.xlabel('Date', fontsize=14)
mp.ylabel('Closing Price', fontsize=14)
# 设置刻度定位器
import matplotlib.dates as md
ax = mp.gca()
ax.xaxis.set_major_locator(
    md.WeekdayLocator(byweekday=md.MO))
ax.xaxis.set_minor_locator(md.DayLocator())
ax.xaxis.set_major_formatter(
    md.DateFormatter('%Y/%m/%d'))
# 把dates转成适合mp绘图的格式
dates = dates.astype(md.datetime.datetime)
mp.plot(dates, closing_prices, color='dodgerblue',
        linewidth=2, linestyle='--',
        label='closing', alpha=0.3)

# 绘制趋势线
trend_points = (highest_prices + lowest_prices +
                closing_prices) / 3
mp.scatter(dates, trend_points, color='orangered',
           s=60, label='Trend Point')
# 整理A与B，求得k与b
times = dates.astype('M8[D]').astype('i4')
A = np.column_stack((times, np.ones_like(times)))
B = trend_points
x = np.linalg.lstsq(A, B)[0]
# y = kx + b 求得每天的趋势线上的值
trend_line = x[0] * times + x[1]
mp.plot(dates, trend_line, color='orangered',
        label='Trend Line')
print(x[0])

mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
