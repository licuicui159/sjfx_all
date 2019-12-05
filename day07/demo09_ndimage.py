# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import scipy.misc as sm
import scipy.ndimage as sn
import matplotlib.pyplot as mp
# 读取文件
original = sm.imread('../da_data/lily.jpg', True)

# pip3 install opencv-python==3.4.1 -i http://xx
# import cv2
# cv2.imread('../xxx.jpg', 0)

# 高斯模糊
median = sn.median_filter(original, 21)
# 角度旋转
rotate = sn.rotate(original, 45)
# 边缘识别
prewitt = sn.prewitt(original)
mp.figure('Image', facecolor='lightgray')
mp.subplot(221)
mp.title('Original', fontsize=16)
mp.axis('off')
mp.imshow(original, cmap='gray')
mp.subplot(222)
mp.title('Median', fontsize=16)
mp.axis('off')
mp.imshow(median, cmap='gray')
mp.subplot(223)
mp.title('Rotate', fontsize=16)
mp.axis('off')
mp.imshow(rotate, cmap='gray')
mp.subplot(224)
mp.title('Prewitt', fontsize=16)
mp.axis('off')
mp.imshow(prewitt, cmap='gray')
mp.tight_layout()
mp.show()
