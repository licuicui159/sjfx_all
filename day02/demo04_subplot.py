# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo04_subplot.py
"""
import numpy as np
import matplotlib.pyplot as mp

mp.figure('Subplot', facecolor='lightgray')

for i in range(1, 10):
    mp.subplot(3, 3, i)
    mp.text(0.5, 0.5, i, ha='center', va='center',
            size=36, alpha=0.5)
    mp.xticks([])
    mp.yticks([])
    mp.tight_layout()

mp.show()
