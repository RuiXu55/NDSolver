#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt

colors=['r','k','g']
if __name__ == '__main__':
  for i in range(0,3):
    data = np.load('out'+str(i+1)+'.npy').item()
    plt.plot(data['wave_k'],data['fzeta'].imag,lw=2,c=colors[i],linestyle='-',label='imag[%d]'%i)
    plt.plot(data['wave_k'],data['fzeta'].real,lw=2,c=colors[i],linestyle='--',label='real[%d]'%i)
  plt.ylim([0,0.8])
  plt.xlabel('$kd_i$')
  plt.ylabel('$\omega/\Omega_i$')
  plt.legend()
  plt.savefig('Para-Lazar-fig6d.pdf',format='pdf')
  plt.show()
