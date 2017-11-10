#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Specify command line arguments')
parser.add_argument('-i','--input', help='Input file name',required=False)
args = parser.parse_args()

if __name__ == '__main__':
  if not args.input :
    data = np.load('output.npy').item()
  else:
    data = np.load(args.input+'.npy').item()

  #levels = [-40,0,40,80,120,160,200]
  X, Y = np.meshgrid(data['wave_k'],data['theta'])
  cset1 = plt.contourf(X, Y, data['fzeta'].imag)
  plt.xlabel('$kd_i$',fontsize=26)
  plt.ylabel(r'$\theta$',fontsize=26)
  plt.colorbar()
  plt.rcParams['font.size']=18
  #plt.savefig('tmp1.pdf',format='pdf',bbox_inches='tight')
  plt.show()
