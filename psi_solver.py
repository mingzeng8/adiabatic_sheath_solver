import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kn

dr = 0.001
Nr = 10000
dr2 = dr * dr
half_dr = dr/2

r = np.zeros(Nr)
# int_n_r_dr, int_n_v_r_dr, psi, v, n defined on the same grid points as r
int_n_r_dr = np.zeros(Nr)
int_n_v_r_dr = np.zeros(Nr)
psi = np.zeros(Nr)
v = np.zeros(Nr)
n = np.zeros(Nr)

# the lower and upper limit of psi for diverging detection
psi_too_negative = -0.9
psi_too_positive = 10.

def get_v(psi): return 2./(1.+np.square(1.+psi))-1.

def psi_vs_r(rc, psic):
    '''
        Solve psi using the parameter rc, and the assumed initial condition psi(r=rc)=psic
    '''
    global r
    r = rc-half_dr + np.arange(Nr)*dr
    rc2 = rc*rc
    half_rc2_over_one_minus_vc = rc2/2/(1-get_v(psic))
    # intialize
    psi[0] = psic + (rc2-r[0]*r[0])/4
    v[0] = get_v(psi[0])
    psi[1] = psi[0] - rc * half_dr
    v[1] = get_v(psi[1])
    # iteration
    for i in range(1,Nr-1):
        # constants for reuse
        pre_n_r = n[i-1]*r[i-1]
        half_dr_over_r = dr/r[i]/2
        
        n[i] = (r[i]*r[i]/2 - (1-v[i])*half_rc2_over_one_minus_vc - int_n_r_dr[i-1] + v[i]*int_n_v_r_dr[i-1] - half_dr*pre_n_r*(1-v[i-1]*v[i])) / (half_dr*r[i]*(1-v[i]*v[i]))
        psi[i+1] = ((1-n[i]*(1-v[i]))*dr2 + (1-half_dr_over_r)*psi[i-1] - 2*psi[i]) / (-1-half_dr_over_r)
        # prevent diverging
        if psi[i+1]<psi_too_negative or psi[i+1]>psi_too_positive: break
        v[i+1] = get_v(psi[i+1])
        int_n_r_dr[i] = int_n_r_dr[i-1] + (pre_n_r+n[i]*r[i])*half_dr
        int_n_v_r_dr[i] = int_n_v_r_dr[i-1] + (pre_n_r*v[i-1]+n[i]*r[i]*v[i])*half_dr
    # ending
    n[i+1] = n[i]
    # if not diverging, return psi[-1], otherwise return psi where diverging behavior is found
    return psi[i+1]

# The iteration limits for the search
iter_max = 30
tolerance = 1e-7

def psic_guess(rc):
    '''
        return the guess of psic based on the theory
    '''
    #psic = np.piecewise(rc, [rc<0.3], [lambda rc: rc**2/2*kn(0, rc), lambda rc: -0.013*rc*rc+0.364*rc-0.044])
    if rc<0.3: psic = rc**2/2*kn(0, rc)
    else: psic = -0.013*rc*rc+0.364*rc-0.044
    return psic

def shooting_psi_vs_r(rc):
    '''
        Use shooting method and bisection method to find psic, which satisfies psi(r -> +infty)=0.
    '''
    guess = psic_guess(rc)
    # initial left boundary of psic
    psicl = guess * 0.99
    # initial right boundary of psic
    psicr = guess * 1.01
    # adjust psicl, if last_psi_l is not negative
    for j in range(iter_max):
        last_psi_l = psi_vs_r(rc, psicl)
        if last_psi_l<0: break
        psicl *= 0.9
    if last_psi_l>0: print('Warning! Suitable initial value of psicl is not found!')
    # adjust psicr, if last_psi_r is not positive
    for j in range(iter_max):
        last_psi_r = psi_vs_r(rc, psicr)
        if last_psi_r>0: break
        psicr *= 1.1
    if last_psi_r<0: print('Warning! Suitable initial value of psicr is not found!')
    # bisection iteration
    for j in range(iter_max):
        psicm = (psicl+psicr)/2
        last_psi_m = psi_vs_r(rc, psicm)
        if last_psi_m<0: psicl = psicm
        else: psicr = psicm
        if (psicr-psicl)<tolerance: break
    if (psicr-psicl)>tolerance: print('Warning! Iteration max is reached for rc = {}!'.format(rc))
    # return the psic value that makes the last psi slightly positive
    return psicr

def psic_vs_rc():
    # Change here to set the range and number of rc to be solved
    rc = np.linspace(0.1, 10., 8)
    
    psic = np.zeros_like(rc)
    for i in range(len(rc)):
        psic[i] = shooting_psi_vs_r(rc[i])
    return rc, psic

if '__main__' == __name__:
    rc, psic = psic_vs_rc()
    print('rc = ', rc)
    print('psic = ', psic)
    plt.plot(rc, psic, 'ks', label='Numerical integral')
    plt.xlabel('$r_c$')
    plt.ylabel('$\\psi_c$')
    plt.legend()
    plt.show()
    
