#!/usr/bin/env python3.x
# -*- coding: UTF-8 -*-

'''
 Public functions related to plasma \zeta function
 Written by Rui Xu May 2017
'''
import sys
import numpy as np
import scipy.special as sp

# Plasma dispersion function
def Z(ze): 
  z = complex(0,1.0)*np.sqrt(np.pi)*sp.wofz(ze)
  return z

'''
Modified dispersion function for k-distribution
using the summation formula in Summers et al 1994
Z_k(ze) = -(k-1/2)/(2*k^3/2) k!/(2*k)! *sum of
          (kappa+l)!/l! i^(k-l)*(2/(z/sqrt(k)+i) 
over l = 0 to k
due to numerical reasons the appearing factorials are combined to:
k!/(2*k)! (k+l)!/l! = (l+1)*...*(k+l) /(k+1)*...*(k+k)
WARNING! ONLY WORK FOR INTEGER k
'''

def Zk(ze,k):
  zk = complex(0.,0.)
  for l in range(0,k+1):
    fac_ratio = 1.0 
    for r in range(1,k+1):
      fac_ratio *= (l+r)/(k+r)

    zk += fac_ratio*(1j)**(k-l)*(2./(ze/np.sqrt(1.*k)+1j))**(k-l+1)
  return  -(k-0.5)/(2.*k**(1.5))*zk

'''modified 'kappa' dispersion function for parallel propagation
   Lazar et. al. 2008
'''
def Zk_para(ze,kappa):
  return (1.+ ze**2/kappa)*Zk(ze,kappa)+ze/kappa*(1.-0.5/kappa)


# Calculate the paralle integral Z_n(zeta)
def p(ze,n):
  w  = 1.0 + ze*Z(ze)
  if   n==0:
    z = Z(ze)
  elif n==1:
    z = w
  elif n==2:
    z = ze*w
  elif n==3:
    z = (1.+2.*ze**2*w)/2.0
  elif n==4:
    z = ze*(1.+2.*ze**2*w)/2.0
  else:
    sys.exit("Error Message:Wrong n for zeta function!")
  return z

# This section Calculate the derivative of parallel integral
# especially dp(ze,0) = Z'(ze)
def dp(ze,n):
  w  = 1.0 + ze*Z(ze)
  dz = -2.0*w
  dw = Z(ze)+ze*dz
  if   n==0:
    dp  = dz
  elif n==1:
    dp  = dw
  elif n==2:
    dp  = w + ze*dw
  elif n==3:
    dp  = 2.*ze*w+ze**2*dw
  elif n==4:
    dp  = (1.+2.*ze**2*w)/2.0+\
          ze*(2*ze*w+ze**2*dw)
  else:
    sys.exit("Error: Wrong n for zeta function derivative!")
  return dp

'''
Derivative of Bessel function of the first kind using 
dJn/dz = 1/2*(J(n-1)- J(n+1))
python module function : sp.jvp(n,ze)
'''
def dJn(ze,n):
  return 0.5*(sp.jv(n-1,ze)-sp.jv(n+1,ze))


'''
# Derivative of exponentially scaled modified 
# Bessel function 
# sp.ive(n,a_s)= I_n(a_s)*exp(-a_s)
# i means ith derivative of the function
'''
def dive(n,a,i):
  a = float(a)
  # No perpendicular component
  if a ==0:
    if   i == 1:
      fz = 0.5*(sp.ive(n-1,a)+sp.ive(n+1,a))-sp.ive(n,a)
    elif i == 2:
      fz = 0.25*sp.ive(n-2,a)-sp.ive(n-1,a)+1.5*sp.ive(n,a)\
           -sp.ive(n+1,a)+0.25*sp.ive(n+2,a)
    else:
      sys.exit("Error:i should be 1 or 2 for dive function!")
   
  # with perpendicular component
  else:
    if   i == 1:
      fz = (n/a-1)*sp.ive(n,a)+sp.ive(n+1,a)
    elif i == 2:
      fz = ((n**2-n)/a**2-2*n/a+1.)*sp.ive(n,a)+\
           ((2*n+1.0)/a-2)*sp.ive(n+1,a)+sp.ive(n+2,a)
    else:
      sys.exit("Error:i should be 1 or 2 for dive function!")
  return fz
 

if __name__ == '__main__':
  print ('Public functions related to plasma \zeta function')
