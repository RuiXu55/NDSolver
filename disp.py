import sys
import scipy
import logging
import zetaf as f
import numpy as np
import scipy.special as sp
from scipy.integrate import quad

# check whether the summation over n is already accurate enough
def check_eps(epsi,eps,N,err):
  zipped = zip([0,0,0,1,1,2],[0,1,2,1,2,2])
  if ((N<=5) or sum(abs((epsi[i,j].real-eps[i,j].real)/epsi[i,j].real)>err for i,j in zipped) or \
   sum(abs((epsi[i,j].imag-eps[i,j].imag)/epsi[i,j].imag)>err for i,j in zipped)):
    var = False
  else:
    var = True
  for i,j in zipped:
    epsi[i,j] = eps[i,j]
  if(N>=10):
    var = True
  return epsi,var


'''
intgrand for kappa-distribution dispersion
'''
def intgrand(s,*args):
  h1,h2,kappa,n,case = args
  Zk = f.Zk(h1/np.sqrt(s),int(kappa+1))
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

def complex_quadrature(intgrand,zh,jh,kappa,n,case):
  data = (zh,jh,kappa,n,case)
  def real_func(x,*args):
    return scipy.real(intgrand(x,*args))
  def imag_func(x,*args):
    return scipy.imag(intgrand(x,*args))
  real_integral = quad(real_func, 1, np.inf, args=data)
  imag_integral = quad(imag_func, 1, np.inf, args=data)
  return (real_integral[0] + 1j*imag_integral[0])

''' General dispersion relation for bi-maxwellian & kappa distribution '''
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
    kappa      = int(p['kappa'][m])

    ''' chiX shortcut for long terms '''
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
      while(True):
        add_eps = np.zeros((3,3),dtype=complex)
        for n in range(-N,N+1):
          # this piece is to avoid unnessary function evaluation
          if(lab and abs(n)<=N-1):
            pass
          else:
            eta   = beta_ratio*omega-(beta_ratio-1.)*n*mu*q
            zh    = np.sqrt((2.*kappa+2.)/(2.*kappa-3.))*(omega-n*q*mu)/\
                    (np.sqrt(beta_para*mu)*k*np.cos(theta))*np.sqrt(dens)
            jh    = k*np.sin(theta)/q*np.sqrt((2.*kappa-3.)*beta_perp/2./mu/dens)
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
        add_eps += epsi

        # check if we should increase N, and copy add_eps value to epsi
        epsi,var = check_eps(epsi,add_eps,N,p['eps_error'][0])
        if(var):
          logger.debug("sp[%d], n=%d satisfies constraint!\n",m,N)
          break
        lab = 1
        N += 1
        logger.debug("sp[%d], Increase N to =%d!\n",m,N)
      epsilon += add_eps

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
            add_eps[0,1] += 1j*chi*n*(f.dive(n,Lam,1)-sp.ive(n,Lam))*eta*f.Z(zeta)
            # below add by Rui based on DSHARK
            add_eps[0,1] += 1j*mu*q**2*n*(f.dive(n,Lam,1)-sp.ive(n,Lam))*(beta_ratio-1.)
            add_eps[0,2] += -mu*dens**2*q**3/beta_perp/(k**2*np.sin(theta)*np.cos(theta))*eta*\
                n*sp.ive(n,Lam)*f.dp(zeta,0)
            add_eps[1,1] += chi*(n**2*sp.ive(n,Lam)/Lam-2.*Lam*(f.dive(n,Lam,1)-sp.ive(n,Lam)))*eta*f.Z(zeta)
            add_eps[1,2] += 1j/2.*dens*q*np.tan(theta)*eta*(f.dive(n,Lam,1)-sp.ive(n,Lam))*f.dp(zeta,0)
            add_eps[2,2] += -dens**2*q**2*(omega-n*mu*q)/beta_perp/(k*np.cos(theta))**2*eta*\
               sp.ive(n,Lam)*f.dp(zeta,0)
        add_eps += epsi
        # check if we should increase N
        epsi,var = check_eps(epsi,add_eps,N,p['eps_error'][0])
        if(var):
          logger.debug("sp[%d], n=%d satisfies constraint!\n",m,N)
          break
        lab = 1
        N += 1
        logger.debug("sp[%d], Increase N to =%d!\n",m,N)
      epsilon += add_eps

  ''' calculate det '''
  disp_det = (epsilon[0,0]-(k*np.cos(theta))**2)*(epsilon[1,1]-k**2)*(epsilon[2,2]-\
    (k*np.sin(theta))**2)+2.0*epsilon[0,1]*epsilon[1,2]*(epsilon[0,2]+\
    k**2*np.sin(theta)*np.cos(theta))-(epsilon[1,1]-k**2)*(epsilon[0,2]+\
    k**2*np.sin(theta)*np.cos(theta))**2+(epsilon[0,0]-(k*np.cos(theta))**2)*\
    epsilon[1,2]**2+(epsilon[2,2]-(k*np.sin(theta))**2)*epsilon[0,1]**2

  disp_det /= omega**p['exp'][0]
  logger.debug("disp_det = %e+%ei \n",disp_det.real,disp_det.imag)
  return (disp_det.real,disp_det.imag)

""" dispersion relation for parallal propagation """
def det_para(z,*data):
  args,p,k = data

  omega    = z[0] +1j*z[1]  # omega/Omega_ci
  disp_det = (p['delta'][0]*omega)**2-k**2
  pol = p['polarization'][0]
  for m in range(int(p['Nsp'][0])):
    beta_perp  = p['beta_perp'][m]
    beta_para  = p['beta_para'][m]
    beta_ratio = beta_perp/beta_para
    kap        = int(p['kappa'][m])
    ak         = np.sqrt(2.*kap/(2.*kap-3.))
    dens       = p['dens'][m]
    mu         = p['mu'][m]
    q          = p['q'][m]

    ze         = ak*(omega+(-1)**pol*mu*q)/(k*np.sqrt(beta_para*mu))
    ze0        = ak*omega/(k*np.sqrt(beta_para*mu))
    disp_det  += dens*mu*q**2*(ze0*f.Zk_para(ze,kap)+\
            (beta_ratio-1.)*(1.+ze*f.Zk_para(ze,kap)))

    ''' from summers 1994'''
    #ze         = ak*(omega+mu*q)/k/np.sqrt(beta_para)*((kap-1)/kap)**0.5/np.sqrt(mu)
    #ze0        = ak*omega/k/np.sqrt(beta_para)/np.sqrt(mu)
    #disp_det  += dens*mu*q**2*(ze0*(kap/(kap-1.5))*((kap-1)/kap)**1.5*f.Zk(ze,kap-1)+\
    #        (beta_ratio-1.)*(1.+(kap-1.)/(kap-1.5)*ze*f.Zk(ze,kap-1)))
  return (disp_det.real,disp_det.imag)
