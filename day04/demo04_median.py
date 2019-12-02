# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo04_median.py  中位数
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
        linewidth=2, linestyle='--', label='closing')

# 求出算数平均数
mean = np.mean(closing_prices)
mean = closing_prices.mean()
mp.hlines(mean, dates[0], dates[-1],
          color='orangered', label='Mean(CP)')

# VWAP
vwap = np.average(closing_prices, weights=volumes)
mp.hlines(vwap, dates[0], dates[-1],
          color='green', label='VWAP')

# TWAP
times = np.linspace(1, 7, 30)
twap = np.average(closing_prices, weights=times)
mp.hlines(twap, dates[0], dates[-1],
          color='gold', label='TWAP')

# median
median = np.median(closing_prices)
# 手动计算
sorted_prices = np.msort(closing_prices)
size = sorted_prices.size
median = (sorted_prices[int((size - 1) / 2)] +
          sorted_prices[int(size / 2)]) / 2
mp.hlines(median, dates[0], dates[-1],
          color='violet', label='median(CP)')


mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
