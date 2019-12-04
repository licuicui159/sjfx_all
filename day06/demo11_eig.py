# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo11_eig.py 特征值与特征向量
"""
import numpy as np

A = np.mat('1 4 7 9; 2 6 7 9; 3 4 7 8; 4 6 7 9')
print(A)
# 提取特征值与特征向量
eigvals, eigvecs = np.linalg.eig(A)
print(eigvals)
print(eigvecs)

# 干坏事
eigvals[1:] = 0
# 逆向推导原矩阵
S = eigvecs * np.diag(eigvals) * eigvecs.I
print(S)
