# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo04_poly.py
"""
import numpy as np
import matplotlib.pyplot as mp

P = [4, 3, -1000, 1]

x = np.linspace(-20, 20, 1000)
y = np.polyval(P, x)

# 求导
Q = np.polyder(P)
xs = np.roots(Q)
print(xs)
ys = np.polyval(P, xs)

mp.scatter(xs, ys, s=100, color='red', marker='o')
mp.plot(x, y)
mp.show()
