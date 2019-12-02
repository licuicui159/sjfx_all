# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo08_attrs.py
"""
import numpy as np
a = np.array([[1 + 1j, 2 + 4j, 3 + 7j],
              [4 + 2j, 5 + 5j, 6 + 8j],
              [7 + 3j, 8 + 6j, 9 + 9j]])
print(a.shape)
print(a.ndim)
print(a.dtype)
print(a.itemsize)
print(a.nbytes)
print(a.real)
print(a.imag)
print(a.T)
print([x for x in a.flat])
