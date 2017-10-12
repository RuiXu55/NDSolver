#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

data = np.load('output.npy').item()
plt.plot(data['wave_k'],data['fzeta'].imag,lw=2,label='imag')
plt.plot(data['wave_k'],data['fzeta'].real,lw=2,label='real')
plt.xlabel('$kd_i$')
plt.ylabel('$\omega/\Omega_i$')
plt.legend()
plt.show()
