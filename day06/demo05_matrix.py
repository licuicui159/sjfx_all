# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_matrix.py
"""
import numpy as np

ary = np.arange(1, 9).reshape(2, 4)
print(ary, type(ary))
# matrix
m1 = np.matrix(ary)
print(m1, type(m1))
m2 = np.mat(ary)
print(m2, type(m2))
# 字符串拼块规则
print(np.mat('1 2 3; 4 5 6'))

# 矩阵乘法
e = np.mat('1 2 6; 3 5 7; 4 8 9')
print(e * e)
a = np.array(e)
print(a * a)

# 逆矩阵
print('-' * 45)
print(e)
print(e.I)
print(e * e.I)

# 应用题
A = np.mat('3 3.2; 3.5 3.6')
B = np.mat('118.4; 135.2')
x = np.linalg.lstsq(A, B)[0]
print(x)
x = np.linalg.solve(A, B)   # 解方程组
print(x)
# 矩阵的解法：
x = A.I * B
print(x)

# 斐波那契数列
f = np.mat('1 1; 1 0')
for i in range(1, 30):
    print(f**i, '<- ', i)
