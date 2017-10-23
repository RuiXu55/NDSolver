import sys
import scipy
import logging
import zetaf as f
import numpy as np
import scipy.special as sp
from numpy import linalg as LA
from scipy.integrate import quad


# check whether the summation over n is already accurate enough
def check_eps(epsi,add_eps,err,lab):
  zipped = zip([0,0,0,1,1,2],[0,1,2,1,2,2])
  if ((lab==0) or sum(abs(add_eps[i,j].real/epsi[i,j].real)>err for i,j in zipped) or \
   sum(abs(add_eps[i,j].imag/epsi[i,j].imag)>err for i,j in zipped)):
    return True,False
  else:
    return True,True

'''
intgrand for kappa-distribution dispersion
'''
def intgrand(s,*args):
  h1,h2,kappa,n,case = args
  Zk = f.Zk(h1/np.sqrt(s),kappa+1)
  # in epsilon[0,0], epsilon[0,2],epsilon[2,2]
  if  (case==0):
    return sp.jv(n,h2*np.sqrt(s-1.))**2/(s**(kappa+2.))*Zk
  # in epsilon[1,1]
  elif(case==1):
    return (s-1.)*sp.jvp(n,h2*np.sqrt(s-1.))**2/(s**(kappa+2.))*Zk
  # in epsilon[0,1], epsilon[1,2]
  elif case==2:
    return np.sqrt(s-1.)*sp.jv(n,h2*np.sqrt(s-1.))*sp.jvp(n,h2*np.sqrt(s-1.))\
           /(s**(kappa+2.))*Zk
  else:
    sys.exit("FATAL ERROR in function intgrand: case does not exist! \n")

''' integrate complex function along real-axis '''
def complex_quadrature(intgrand,zh,jh,kappa,n,case):
  data = (zh,jh,kappa,n,case)
  def real_func(x,*args):
    return scipy.real(intgrand(x,*args))
  def imag_func(x,*args):
    return scipy.imag(intgrand(x,*args))
  int_real = quad(real_func, 1, np.inf, args=data)
  int_imag = quad(imag_func, 1, np.inf, args=data)
  return (int_real[0] + 1j*int_imag[0])

''' 
General dispersion relation for bi-maxwellian & kappa distribution
adopted from Astfalk et. al. 2015
'''
def det(z,*data):
  args,p,k = data
  logging.basicConfig(level=args.loglevel or logging.INFO)
  logger = logging.getLogger(__name__)

  omega  = z[0] +1j*z[1]  # omega/Omega_ci
  theta  = p['theta'][0]*np.pi/180.
  logger.debug("omega guess = %e+%ei \n",omega.real,omega.imag)

  # epsilon is renormalized: epsilon = epsilon*(v_A^2/c^2*omega^2)
  epsilon  = np.zeros((3,3),dtype=complex)
  epsilon += (p['delta'][0]*omega)**2*np.identity(3)

  """ Iterate over species """
  for m in range(0,int(p['Nsp'][0])):
    beta_perp  = p['beta_perp'][m]
    beta_para  = p['beta_para'][m]
    beta_ratio = beta_perp/beta_para
    dens       = p['dens'][m]
    mu         = p['mu'][m]
    q          = p['q'][m]
    kappa      = int(p['kappa'][m]) # right now only integer kappa is supported

    ''' chiX shortcuts for long terms '''
    chi        = dens**1.5*q**2*np.sqrt(mu/beta_para)/(k*np.cos(theta))
    chi0       = (beta_ratio-1.)*mu*dens*q**2

    ''' use kappa dispersion relation '''
    if(kappa<p['kappa_limit'][0]):

      ''' first add non-iterative term '''
      epsilon[0,0] += chi0
      epsilon[1,1] += chi0
      epsilon[2,2] += chi0*np.tan(theta)**2
      epsilon[2,2] += 2.*omega**2*(2.*kappa-1.)/(2.*kappa-3.)*dens**2\
		      *q**2/beta_para/(k*np.cos(theta))**2
      epsilon[0,2] += -chi0*np.tan(theta)  # typo?
      
      N     = int(p['N'][0])
      lab   = 0
      epsi     = np.zeros((3,3),dtype=complex)
      while True :
        add_eps = np.zeros((3,3),dtype=complex)
        for n in range(-N,N+1):
          # this piece is to avoid unnessary function evaluation
          if(lab and abs(n)<=N-1):
            pass
          else:
            eta   = beta_ratio*omega-(beta_ratio-1.)*n*mu*q # shortcut
            zh    = np.sqrt((2.*kappa+2.)/(2.*kappa-3.))*(omega-n*q*mu)/\
                    (np.sqrt(beta_para*mu)*k*np.cos(theta))*np.sqrt(dens)
            jh    = k*np.sin(theta)*np.sqrt((2.*kappa-3.)*beta_perp/2./mu/dens)/q
            intxx = complex_quadrature(intgrand,zh,jh,kappa,n,0)
            intyy = complex_quadrature(intgrand,zh,jh,kappa,n,1)
            intxy = complex_quadrature(intgrand,zh,jh,kappa,n,2)

            add_eps[0,0] += 4.*np.sqrt(2.)*mu**1.5*dens**2.5*q**4*(kappa-0.5)*((kappa+1.)/(2.*kappa-3.))**1.5\
              /(beta_perp*(k*np.sin(theta))**2)/(np.sqrt(beta_para)*k*np.cos(theta))*n**2*eta*intxx
            add_eps[1,1] += 2.*np.sqrt(2.)*chi*(kappa-0.5)/np.sqrt(2.*kappa-3.)*(kappa+1.)**1.5*eta*intyy
            add_eps[2,2] += 4.*np.sqrt(2.)/np.sqrt(mu)*dens**2.5*q**2*(kappa-0.5)*((kappa+1.)/(2.*kappa-3.0))**1.5\
              /beta_perp/np.sqrt(beta_para)/(k*np.cos(theta))**3*eta*(omega-n*mu*q)**2*intxx
            add_eps[0,1] += 4j*mu*dens**2*q**3/np.sqrt(beta_perp*beta_para)/(k**2*np.cos(theta)*np.sin(theta))*\
              (kappa-0.5)/(2.*kappa-3.)*(kappa+1.)**1.5*n*eta*intxy
            add_eps[0,2] += 4.*np.sqrt(2.)*np.sqrt(mu)*dens**2.5*q**3*(kappa-0.5)*((kappa+1)/(2.*kappa-3.))**1.5/\
              beta_perp/(k*np.sin(theta))/np.sqrt(beta_para)/(k*np.cos(theta))**2*n*eta*(omega-n*mu*q)*intxx
            add_eps[1,2] += -4j*dens**2*q**2*(kappa-0.5)/(2.*kappa-3.)*(kappa+1.)**1.5/np.sqrt(beta_perp*beta_para)\
              /(k*np.cos(theta))**2*eta*(omega-n*mu*q)*intxy

        # check if we should increase N, and copy add_eps value to epsi
        lab,var = check_eps(epsi,add_eps,p['eps_error'][0],lab)
        if(var):
          logger.debug("sp[%d], n=%d satisfies constraint!\n",m,N)
          break
        epsi += add_eps
        N    += 1
        logger.debug("sp[%d], Increase N to =%d!\n",m,N)
      epsilon += epsi

    else:
      '''use bi-maxwellian dispersion'''
      Lam        = (k*np.sin(theta))**2*beta_perp/(2.*q**2*mu*dens)
      # add non-iterative term
      epsilon[0,0] += chi0
      epsilon[1,1] += chi0

      N     = int(p['N'][0])
      lab   = 0
      epsi     = np.zeros((3,3),dtype=complex)
      while(True):
        add_eps = np.zeros((3,3),dtype=complex)
        for n in range(-N,N+1):
          # this piece is to avoid unnessary function evaluation
          if(lab and abs(n)<=N-1):
            pass
          else:
            eta  = beta_ratio*omega-(beta_ratio-1.)*n*mu*q
            zeta = (omega-n*mu*q)*np.sqrt(dens)/(np.sqrt(beta_para*mu)*k*np.cos(theta))
            add_eps[0,0] += chi*n**2*sp.ive(n,Lam)/Lam*eta*f.Z(zeta)
            add_eps[0,1] += 1j*chi*n*(f.dIne(n,Lam)-sp.ive(n,Lam))*eta*f.Z(zeta)
            add_eps[0,2] += -mu*dens**2*q**3/beta_perp/(k**2*np.sin(theta)*np.cos(theta))*eta*\
                n*sp.ive(n,Lam)*f.dp(zeta,0)
            add_eps[1,1] += chi*(n**2*sp.ive(n,Lam)/Lam-2.*Lam*(f.dIne(n,Lam)-sp.ive(n,Lam)))*eta*f.Z(zeta)
            add_eps[1,2] += 1j/2.*dens*q*np.tan(theta)*eta*(f.dIne(n,Lam)-sp.ive(n,Lam))*f.dp(zeta,0)
            add_eps[2,2] += -dens**2*q**2*(omega-n*mu*q)/beta_perp/(k*np.cos(theta))**2*eta*\
               sp.ive(n,Lam)*f.dp(zeta,0)
        # check if we should increase N
        lab, var = check_eps(epsi,add_eps,p['eps_error'][0],lab)
        if(var):
          logger.debug("sp[%d], n=%d satisfies constraint!\n",m,N)
          break
        N     += 1
        epsi  += add_eps
        logger.debug("sp[%d], Increase N to =%d!\n",m,N)
      epsilon += epsi
  epsilon[1,0]  = -epsilon[0,1]
  epsilon[2,0]  = epsilon[0,2]
  epsilon[2,1]  = -epsilon[1,2]
  epsilon[0,0] += -(k*np.cos(theta))**2
  epsilon[1,1] += -k**2
  epsilon[2,2] += -(k*np.sin(theta))**2
  epsilon[0,2] += k**2*np.sin(theta)*np.cos(theta)
  epsilon[2,0] += k**2*np.sin(theta)*np.cos(theta)
  ''' calculate determinant '''
  disp_det = LA.det(epsilon)/omega**p['exp'][0]

  ''' calculate polarization 
  val = np.ones(3,dtype =complex)    # eigenvalue 
  vec = np.ones((3,3),dtype=complex) # eigenvectors 
  val, vec = LA.eig(epsilon)
  pol = 1j*vec[1]/vec[0]*omega.real/abs(omega.real)
  '''
  logger.debug('for omega=',omega,"disp_det = %e+%ei \n",disp_det.real,disp_det.imag)
  return (disp_det.real,disp_det.imag)

""" 
dispersion relation for parallal propagation 
from Lazar et al. 2011 and Gary & Madland 1985   
"""
def det_para(z,*data):
  args,p,k = data
  logging.basicConfig(level=args.loglevel or logging.INFO)
  logger = logging.getLogger(__name__)

  omega    = z[0] +1j*z[1]  # omega/Omega_ci
  pol      = p['polarization'][0]
  disp_det = (p['delta'][0]*omega)**2-k**2
  ''' bi-kappa from Lazar et. al. 2011 '''
  for m in range(int(p['Nsp'][0])):
    beta_perp = p['beta_perp'][m]
    beta_para = p['beta_para'][m]
    kap       = int(p['kappa'][m])
    dens      = p['dens'][m]
    mu        = p['mu'][m]
    q         = p['q'][m]

    ze        = np.sqrt(2.*kap/(2.*kap-3.))*(omega+(-1)**pol*mu*q)/(k*np.sqrt(beta_para*mu))
    ze0       = np.sqrt(2.*kap/(2.*kap-3.))*omega/(k*np.sqrt(beta_para*mu))
    disp_det += dens*mu*q**2*(ze0*f.Zk_para(ze,kap)+\
                    (beta_perp/beta_para-1.)*(1.+ze*f.Zk_para(ze,kap)))
  logger.debug('for omega=',omega,"disp_det = %e+%ei \n",disp_det.real,disp_det.imag)
  return (disp_det.real,disp_det.imag)

if __name__ == '__main__':
  print ('dispersion relation dielectric tensor.')
