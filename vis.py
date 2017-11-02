#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Specify command line arguments')
parser.add_argument('-i','--input', help='Input file name',required=False)
args = parser.parse_args()

if __name__ == '__main__':
  if not args.input :
    data = np.load('output.npy').item()
  else:
    data = np.load(args.input+'.npy').item()
  plt.plot(data['wave_k'],data['fzeta'].imag,lw=2,label='Imag')
  plt.plot(data['wave_k'],data['fzeta'].real,lw=2,label='Real')
  plt.xlabel('$kd_i$')
  plt.ylabel('$\omega/\Omega_i$')
  plt.rcParams['font.size'] = 16
  #plt.savefig('sample.pdf',format='pdf')
  plt.legend()
  plt.show()
