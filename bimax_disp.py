import sys
import zetaf as f
import numpy as np
import logging 
import scipy.special as sp

# check whether the summation over n is accurate enough
def check_eps(epsi,eps,N,err):
  if((N<=4) or (abs((epsi[0,0].real-eps[0,0].real)/epsi[0,0].real)>err) or\
    (abs((epsi[0,0].imag-eps[0,0].imag)/epsi[0,0].imag)>err) or\
    (abs((epsi[0,1].real-eps[0,1].real)/epsi[0,1].real)>err) or\
    (abs((epsi[0,1].imag-eps[0,1].imag)/epsi[0,1].imag)>err) or\
    (abs((epsi[0,2].real-eps[0,2].real)/epsi[0,2].real)>err) or\
    (abs((epsi[0,2].imag-eps[0,2].imag)/epsi[0,2].imag)>err) or\
    (abs((epsi[1,1].real-eps[1,1].real)/epsi[1,1].real)>err) or\
    (abs((epsi[1,1].imag-eps[1,1].imag)/epsi[1,1].imag)>err) or\
    (abs((epsi[1,2].real-eps[1,2].real)/epsi[1,2].real)>err) or\
    (abs((epsi[1,2].imag-eps[1,2].imag)/epsi[1,2].imag)>err) or\
    (abs((epsi[2,2].real-eps[2,2].real)/epsi[2,2].real)>err) or\
    (abs((epsi[2,2].imag-eps[2,2].imag)/epsi[2,2].imag)>err) ):
    var = False
  else:
    var = True
  epsi[0,0] = eps[0,0]
  epsi[1,1] = eps[1,1]
  epsi[2,2] = eps[2,2]
  epsi[0,1] = eps[0,1]
  epsi[0,2] = eps[0,2]
  epsi[1,2] = eps[1,2]
  return epsi,var

# Bi-Maxwellian dispersion
def det(z,*data):
  args,p,k = data
  logging.basicConfig(level=args.loglevel or logging.INFO)
  logger = logging.getLogger(__name__)

  omega  = complex(z[0],z[1])  # omega/Omega_i
  theta  = p['theta'][0]*np.pi/180.
  j      = complex(0,1.)

  # epsilon is renormalized: epsilon = epsilon*(v_A^2/c^2*omega^2)  
  epsilon  = np.zeros((3,3),dtype=complex)
  epsi     = np.zeros((3,3),dtype=complex)
  epsilon += (p['delta'][0]*omega)**2*np.identity(3)

  """ Iterate over species """
  for m in range(0,int(p['Nsp'][0])):
    beta_perp  = p['beta_perp'][m]
    beta_para  = p['beta_para'][m]
    beta_ratio = beta_perp/beta_para
    dens       = p['dens'][m]
    mu         = p['mu'][m]
    q          = p['q'][m]
    Lam        = (k*np.sin(theta))**2*beta_perp/(2.*q**2*mu*dens)

    # add non-iterative term
    for i in range(0,2):
      epsilon[i,i] += (beta_ratio-1.)*mu*dens*q**2

    N = 4
    while(True):
      add_eps = np.zeros((3,3),dtype=complex)
      for n in range(-N,N+1):
	chi  = dens**1.5*np.sqrt(mu)*q**2/np.sqrt(beta_para)/(k*np.cos(theta))
        zeta = (omega-n*mu*q)/np.sqrt(beta_para)/(k*np.cos(theta))/np.sqrt(mu)*np.sqrt(dens)
	eta  = beta_ratio*omega-(beta_ratio-1.)*n*mu*q
        add_eps[0,0] += chi*n**2*sp.ive(n,Lam)/Lam*eta*f.Z(zeta)
        add_eps[0,1] += j*chi*n*(f.dive(n,Lam,1)-sp.ive(n,Lam))*eta*f.Z(zeta)
        add_eps[0,2] += -mu*dens**2*q**3/beta_perp/(k**2*np.sin(theta)*np.cos(theta))*eta*\
                        n*sp.ive(n,Lam)*f.dp(zeta,0)
        add_eps[1,1] += chi*(n**2*sp.ive(n,Lam)/Lam-2.*Lam*(f.dive(n,Lam,1)-sp.ive(n,Lam)))*eta*f.Z(zeta)
        add_eps[1,2] += j/2.*dens*q*np.tan(theta)*eta*(f.dive(n,Lam,1)-sp.ive(n,Lam))*f.dp(zeta,0)
        add_eps[2,2] += -dens**2*q**2*(omega-n*mu*q)/beta_perp/(k*np.cos(theta))**2*eta*\
                        sp.ive(n,Lam)*f.dp(zeta,0)
      # check if we should increase N
      epsi,var = check_eps(epsi,add_eps,N,p['eps_error'][0])
      if(var):
        logger.debug("sp[%d], n=%d satisfies constraint!\n",m,N)
        break
      N += 1
    epsilon += add_eps

  disp_det = (epsilon[0,0]-(k*np.cos(theta))**2)*(epsilon[1,1]-k**2)*(epsilon[2,2]-\
    (k*np.sin(theta))**2)+2.0*epsilon[0,1]*epsilon[1,2]*(epsilon[0,2]+\
    k**2*np.sin(theta)*np.cos(theta))-(epsilon[1,1]-k**2)*(epsilon[0,2]+\
    k**2*np.sin(theta)*np.cos(theta))**2+(epsilon[0,0]-(k*np.cos(theta))**2)*\
    epsilon[1,2]**2+(epsilon[2,2]-(k*np.sin(theta))**2)*epsilon[0,1]**2

  logger.debug("disp_det = %e+%ei \n",disp_det.real,disp_det.imag)
  return (disp_det.real,disp_det.imag)

