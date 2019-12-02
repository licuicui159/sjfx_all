# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo05_gridspec.py 网格布局
"""
import matplotlib.pyplot as mp
import matplotlib.gridspec as gs

mp.figure('Grid Spec', facecolor='lightgray')
gridSpec = gs.GridSpec(3, 3)
mp.subplot(gridSpec[0, :2])
mp.text(0.5, 0.5, '1', ha='center', va='center',
        size=36, alpha=0.5)
mp.xticks([])
mp.yticks([])
mp.tight_layout()

mp.subplot(gridSpec[:2, 2])
mp.text(0.5, 0.5, '2', ha='center', va='center',
        size=36, alpha=0.5)
mp.xticks([])
mp.yticks([])
mp.tight_layout()

mp.subplot(gridSpec[1, 1])
mp.text(0.5, 0.5, '3', ha='center', va='center',
        size=36, alpha=0.5)
mp.xticks([])
mp.yticks([])
mp.tight_layout()

mp.subplot(gridSpec[1:, 0])
mp.text(0.5, 0.5, '4', ha='center', va='center',
        size=36, alpha=0.5)
mp.xticks([])
mp.yticks([])
mp.tight_layout()

mp.subplot(gridSpec[-1, 1:])
mp.text(0.5, 0.5, '5', ha='center', va='center',
        size=36, alpha=0.5)
mp.xticks([])
mp.yticks([])
mp.tight_layout()

mp.show()
