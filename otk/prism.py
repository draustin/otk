"""Functions for calculating properties of prisms.

Common definitions:

* lamb: wavelength in m
* omega: frequency in rad/s
* n: refractive index
* nlamb: refractive index function of wavelength
* theta_1: incident angle w.r.t. first face normal, increasing away from apex.
* thetap_1: internal refracted angle w.r.t first face normal
* theta_2: incident angle w.r.t. second face normal, increasing away from apex.
* thetap_2: internal refracted angle w.r.t second face normal
* alpha: apex full angle

The angles are defined so that for alpha=0, theta_2=-theta_1. The deflection angle is therefore theta_1+theta_2-alpha
away from the normal.
"""
import numpy as np
from scipy.misc import derivative

def refract(n,theta_1,alpha,return_internals=False):
    """Calculate refraction angle of prism
    
    All angles are in radians. See `optics.prism_pair' for complete definitions.
    
    Args:
        return_internals: whether to return the internal angles
        
    Returns:
        Returns theta_2 if return_internals is False, (thetap_1,thetap_2,theta_2) otherwise
    """
    # All angles w.r.t. associated normal
    thetap_1=np.arcsin(np.sin(theta_1)/n) # Internal angle on first face
    thetap_2=alpha-thetap_1 # Internal angle on second face
    theta_2=np.arcsin(n*np.sin(thetap_2)) # External angle on second face
    if return_internals:
        return thetap_1,thetap_2,theta_2
    else:
        return theta_2

def dtheta_2_dn(alpha,thetap_1,theta_2):
    """Calculate derivative of output angle w.r.t refractive index."""
    return np.sin(alpha)/(np.cos(thetap_1)*np.cos(theta_2)) 
    
def angular_dispersion(alpha,theta_1,**kwargs):
    """Calculate angular dispersion (derivative of angle w.r.t wavelength).
    
    Args:
        kwargs can be either nlamb and lamb, or n and dn_dlamb
        
    Returns:
        dtheta_2_dlamb, the derivative of the output angle w.r.t wavelength
    """
    if 'nlamb' in kwargs:
        nlamb=kwargs['nlamb']
        lamb=kwargs['lamb']
        n=nlamb(lamb)
        dn_dlamb=derivative(nlamb,lamb,lamb/100)
    else:
        n=kwargs['n']
        dn_dlamb=kwargs['dn_dlamb']
    thetap_1,thetap_2,theta_2=refract(n,theta_1,alpha,return_internals=True)
    return dtheta_2_dn(alpha,thetap_1,theta_2)*dn_dlamb

def beam_expansion(alpha,n,theta_1):
    thetap_1,thetap_2,theta_2=refract(n,theta_1,alpha,True)
    factor=np.cos(theta_2)/np.cos(thetap_2)*np.cos(thetap_1)/np.cos(theta_1)
    return factor,thetap_1,thetap_2,theta_2

def minimum_deviation_incident_angle(alpha,n):
    return np.arcsin(np.sin(alpha/2)*n)