# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_pred.py  线性预测
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
        linewidth=2, linestyle='--', label='closing')

# 线性预测
N = 3
pred_prices = np.zeros(closing_prices.size - 2 * N)
for j in range(pred_prices.size):
    A = np.zeros((N, N))
    for i in range(N):
        A[i:] = closing_prices[i + j:i + j + N]
    B = closing_prices[N + j: j + N * 2]
    x = np.linalg.lstsq(A, B)[0]
    pred = B.dot(x)   # pred = d*w0 + e*w1 + f*w2
    pred_prices[j] = pred
print(pred_prices)
mp.plot(dates[2 * N:], pred_prices, 'o-',
        color='orangered', label='predict prices')

mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
