# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_cov.py  协方差
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
# bhp
dates, bhp_closing_prices = np.loadtxt(
    '../da_data/bhp.csv', delimiter=',',
    usecols=(1, 6), dtype='M8[D], f8',
    unpack=True, converters={1: dmy2ymd})
# vale
vale_closing_prices = np.loadtxt(
    '../da_data/vale.csv', delimiter=',',
    usecols=(6,))


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
mp.plot(dates, bhp_closing_prices, color='dodgerblue',
        linewidth=2, label='bhp_closing')
mp.plot(dates, vale_closing_prices, color='orangered',
        linewidth=2, label='vale_closing')

# 协方差
ave_bhp = np.mean(bhp_closing_prices)
ave_vale = np.mean(vale_closing_prices)
dev_bhp = bhp_closing_prices - ave_bhp
dev_vale = vale_closing_prices - ave_vale
cov = np.sum(dev_bhp * dev_vale) / (dev_bhp.size - 1)
print(cov)

# 相关系数
coef = cov / (np.std(bhp_closing_prices) *
              np.std(vale_closing_prices))
print(coef)

# 相关矩阵
m = np.corrcoef(bhp_closing_prices,
                vale_closing_prices)
print(m, m[0, 1])

#      爱情   动作    喜剧
# u1   20     5       10
# u2   5      1       2
# u3   10     20      30

print(np.cov(bhp_closing_prices, vale_closing_prices))

mp.legend()
mp.gcf().autofmt_xdate()
mp.show()
