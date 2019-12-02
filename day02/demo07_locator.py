# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo07_locator.py
"""
import matplotlib.pyplot as mp
locators = [
    'mp.NullLocator()', 'mp.MaxNLocator(nbins=4)',
    'mp.AutoLocator()', 'mp.MultipleLocator(1)']

mp.figure('Locators', facecolor='lightgray')

for i, locator in enumerate(locators):

    mp.subplot(len(locators), 1, i + 1)
    ax = mp.gca()
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0.5))
    mp.yticks([])
    mp.xlim(1, 10)
    # 设置刻度定位器
    ax.xaxis.set_major_locator(eval(locator))
    ax.xaxis.set_minor_locator(mp.MultipleLocator(0.1))

mp.show()
