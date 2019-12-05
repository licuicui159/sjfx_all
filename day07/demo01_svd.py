# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo01_svd.py 奇异值分解
"""
import numpy as np

M = np.mat('1 4 9; 4 7 11')
print(M)

U, sv, V = np.linalg.svd(M, full_matrices=False)
print(U * U.T, '<- U')
print(sv, '<- sv')
print(V, '<- V')

# 推导原矩阵
sv[1] = 0
M2 = U * np.diag(sv) * V
print(M2)
