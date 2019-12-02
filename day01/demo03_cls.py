# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np

data = [('zs', [90, 80, 85], 15),
        ('ls', [92, 81, 83], 16),
        ('ww', [95, 85, 95], 15)]

# 复合数据类型
ary = np.array(data, dtype='U2, 3int32, int32')
print(ary, ary[1], ary[1][2])

# 第二种设置dtype的方式，为字段设置别名
ary = np.array(data, dtype=[('name', 'str_', 2),
                            ('scores', 'int32', 3),
                            ('age', 'int32', 1)])
print(ary, ary[2]['name'])

# 第三种设置dtype的方式
ary = np.array(data, dtype={
    'names': ['name', 'scores', 'age'],
    'formats': ['U2', '3i4', 'int32']})
print(ary, ary[2]['name'])
print(ary['name'])

d = np.array(data, dtype={'name': ('U3', 0),
                          'scores': ('3int32', 16),
                          'age': ('int32', 28)})
print(d[0]['name'], d[0]['scores'], d.itemsize)

# 日期类型
dates = np.array(['2011', '2011-02', '2011-03-01',
                  '2011-04-01 11:11:11', '2012'])
print(dates, dates.dtype)
# dates = dates.astype('datetime64[D]')
dates = dates.astype('M8[D]')
print(dates, dates.dtype)

print(dates[-1] - dates[0])
