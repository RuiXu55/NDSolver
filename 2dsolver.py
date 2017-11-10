#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Rui Xu, Oct. 2017
# Email: ruix@princeton.edu
import os
import sys
import time
import disp
import utils
import logging 
import pol as p
import argparse
import numpy as np
from scipy import interp
import multiprocessing as mp
from scipy.optimize import root
__author__ = 'ruix'

def Loop(pid, args, param, theta, zeta_guess):
  logging.basicConfig(level=args.loglevel or logging.INFO)
  logger = logging.getLogger(__name__)

  wave_k = param['wave_k']
  param['theta'] = [] 
  param['theta'].append(theta)
  fzeta  = np.empty(len(wave_k),dtype=complex)
  for n in range(len(wave_k)):
    data = (args, param, wave_k[n])
    try:
      # use parallel dsp if theta<0.1 degree
      if (abs(param['theta'][0])<1.):
        sol = root(disp.det_para,(zeta_guess.real,zeta_guess.imag), \
            args=data,method='hybr',tol=param['sol_err'][0]) 
        fzeta[n] = complex(sol.x[0],sol.x[1])
      else:
        sol = root(disp.det,(zeta_guess.real,zeta_guess.imag), \
            args=data,method='hybr',tol=param['sol_err'][0]) 
        fzeta[n] = complex(sol.x[0],sol.x[1])
      if (fzeta[n].imag<0):
        param['exp'][0] = 0
      logger.info('pid = %d, wave_k= %1.1f, fzeta = \
              %1.2e+%1.2e',pid,wave_k[n],fzeta[n].real,fzeta[n].imag)
    except ValueError:
      logger.info('ERROR in root finding: wave_k =%f in processer \
              %d',wave_k[n],pid)

    """ extrapolate previous solutions for next guess """
    if(n>3 and n<int(param['ksteps'][0])-1): 
      dk = wave_k[1] -wave_k[0]
      zeta_guess = complex(interp(wave_k[n]+dk,wave_k[n-3:n],\
      fzeta[n-3:n].real),\
      interp(wave_k[n]+dk,wave_k[n-3:n],fzeta[n-3:n].imag))
    else:
      zeta_guess = fzeta[n]
  return fzeta


def main(args):
  """ parse command line arguments"""
  tstart = time.clock()
  if not args.log :
    logging.basicConfig(level=args.loglevel or logging.INFO)
  else:
    logging.basicConfig(filename='log', filemode='w', level=logging.DEBUG)
  logger = logging.getLogger(__name__)

   """ read plasma parameters """
  param = utils.read_param(args)
  
  """ iterate through wavenumber  """
  dk     = (param['kend'][0]-param['kstart'][0])/param['ksteps'][0]
  wave_k = np.empty(int(param['ksteps'][0]))
  for n in range(int(param['ksteps'][0])):
    wave_k[n] = param['kstart'][0]+n*dk
  param['wave_k'] = wave_k
  theta = np.linspace(param['thetamin'][0],param['thetamax'][0],int(param['ntheta'][0]))
  zeta_guess = complex(param['omega_r'][0],param['omega_i'][0])

  """ parallizization, (increase # of threads may not improve peformance)  """
  pool = mp.Pool(len(theta))
  zeta = []
  for i in range(0,len(theta)):
      res = pool.apply_async(Loop,args=(i,param,theta[i],zeta_guess))
      zeta.append(res)
  pool.close()
  pool.join()

   
  """ save results to output file """
  fzeta = []
  for dat in zeta:
    fzeta.append(np.array(dat.get())) 
  param['fzeta'] = np.array(fzeta)
  param['theta'] = theta

  if args.output :
    np.save(args.output+'.npy',param)
  else:
    np.save('output.npy', param)

  tend = time.clock()
  logger.info('\n Total time lapse: %1.2f\n',tend-tstart)
  return  0

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Specify command line arguments')
  parser.add_argument('-i','--input', help='Input file name',required=False)
  parser.add_argument('-o','--output',help='Output file name')
  parser.add_argument('-l','--log', action='store',help='log file')
  parser.add_argument('-v', '--verbose', help='Verbose (debug) logging',
          action='store_const', const=logging.DEBUG,dest='loglevel')
  args = parser.parse_args()
  print (parser.parse_args())
  sys.exit(main(args))

 
 
