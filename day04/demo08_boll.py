# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo08_boll.py  布林带
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
        label='closing', alpha=0.5)

# 加权5日移动平均线
kernel = np.exp(np.linspace(-1, 0, 5))
kernel = kernel[::-1]
kernel /= kernel.sum()
ma53 = np.convolve(closing_prices, kernel, 'valid')
mp.plot(dates[4:], ma53, color='orangered',
        label='MA(5)-3')
# 从第五天开始，计算最近5日的标准差
stds = np.zeros(ma53.size)
for i in range(stds.size):
    stds[i] = closing_prices[i:i + 5].std()
upper = ma53 + 2 * stds
lower = ma53 - 2 * stds
mp.plot(dates[4:], upper, color='dodgerblue',
        label='upper')
mp.plot(dates[4:], lower, color='dodgerblue',
        label='lower')
mp.fill_between(
    dates[4:], lower, upper, lower < upper,
    color='orangered', alpha=0.2)

mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
