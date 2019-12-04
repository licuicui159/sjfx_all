# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo10_sin.py
"""
import numpy as np
import matplotlib.pyplot as mp

x = np.linspace(0, 4 * np.pi, 1000)
y1 = 4 * np.pi * np.sin(x)
y2 = 4 / 3 * np.pi * np.sin(3 * x)

y = np.zeros(x.size)
n = 1000
for i in range(1, n + 1):
    y += 4 / (2 * i - 1) * np.pi * \
        np.sin((2 * i - 1) * x)

mp.grid(linestyle=':')
mp.plot(x, y1, label='y1', alpha=0.1)
mp.plot(x, y2, label='y2', alpha=0.1)
mp.plot(x, y, label='y')
mp.legend()
mp.show()
