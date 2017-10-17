#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Specify command line arguments')
parser.add_argument('-i','--input', help='Input file name',required=False)
args = parser.parse_args()

colors=['r','k','g','c','m']
if __name__ == '__main__':
  for i in range(0,2):
    if not args.input :
      data = np.load('output'+str(i+1)+'.npy').item()
    else:
      data = np.load(args.input+str(i+1)+'.npy').item()
    plt.plot(data['wave_k'],data['fzeta'].imag,lw=2,c=colors[i],linestyle='-',label='imag[%d]'%i)
    plt.plot(data['wave_k'],data['fzeta'].real,lw=2,c=colors[i],linestyle='--',label='real[%d]'%i)
  plt.axis([0.1,0.45,0,0.08])
  plt.xlabel('$kd_i$')
  plt.ylabel('$\omega/\Omega_i$')
  plt.legend()
  plt.savefig('Oblique-Patrick-fig3.pdf',format='pdf')
  plt.show()
