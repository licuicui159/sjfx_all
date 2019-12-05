# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo02_svd2.py 图像特征
"""
import numpy as np
import scipy.misc as sm
import matplotlib.pyplot as mp

# 读取图片   True: 读取灰度图像
img = sm.imread('../da_data/lily.jpg', True)
print(img.shape)

# 提取特征值
img = np.mat(img)
eigvals, eigvecs = np.linalg.eig(img)
print(eigvals.shape, eigvecs.shape)
# 生成原图像
eigvals[50:] = 0
img2 = eigvecs * np.diag(eigvals) * eigvecs.I


# 奇异值分解
U, sv, V = np.linalg.svd(img)
print(U.shape, sv.shape, V.shape)
sv[50:] = 0
img3 = U * np.diag(sv) * V

mp.subplot(221)
mp.imshow(img, cmap='gray')
mp.xticks([])
mp.yticks([])
mp.tight_layout()

mp.subplot(222)
mp.imshow(img2.real, cmap='gray')
mp.xticks([])
mp.yticks([])
mp.tight_layout()

mp.subplot(223)
mp.imshow(V, cmap='gist_rainbow')
mp.xticks([])
mp.yticks([])
mp.tight_layout()

mp.subplot(224)
mp.imshow(img3, cmap='gray')
mp.xticks([])
mp.yticks([])
mp.tight_layout()

mp.show()
