# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo03_fft.py
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

mp.subplot(121)
mp.title('Time Domain')
mp.grid(linestyle=':')
mp.plot(x, y1, label='y1', alpha=0.1)
mp.plot(x, y2, label='y2', alpha=0.1)
mp.plot(x, y, label='y')

# 基于傅里叶变换，拆这个方波，得到一组正弦函数
import numpy.fft as nf
complex_ary = nf.fft(y)
print(complex_ary[0], complex_ary.shape)
# 逆向傅里叶变换，合成原函数
y_copy = nf.ifft(complex_ary).real
mp.plot(x, y_copy, label='y_copy', linewidth=7,
        alpha=0.3)

# 得到傅里叶变换后这组正弦函数的频率数组
freqs = nf.fftfreq(y.size, x[1] - x[0])
print(freqs.shape)
# 得到的freqs(1000,)与fft后的复数数组(1000,)是配套的
# 由此可以得到1000个正弦函数
# 绘制频域图像
mp.subplot(122)
mp.title('Frequency Domain')
pows = np.abs(complex_ary)  # 每个正弦函数的能量
mp.grid(linestyle=':')
mp.plot(freqs[freqs > 0], pows[freqs > 0],
        color='orangered')

mp.show()
