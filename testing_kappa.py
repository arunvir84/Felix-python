# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 12:37:23 2025

@author: adamc
"""
import time
import numpy as np
import math
import scipy.integrate as integrate
import scipy.constants as c
from pylix_modules import pylix as px
from pylix_modules import pylix_dicts as fu
from numba import njit, prange
import matplotlib.pyplot as plt


#max is 10 angrstrom or bohr radius 
start = time.time()

Bohr = 0.52917721067 # in angstrom


def calc_slater_orbitals(z, orbital,r):
    
    
    #ok so now when we calc slater orbital we pass kappa scaled q 
    #r, distance of electron from atomic nucleus#
    #N is normalizing constant 

    delta = np.array(fu.slater_coefficients[z][orbital]['delta'])
    delta = delta / 0.52917721092
       # convert to angstrom 
    
    C= np.array(fu.slater_coefficients[z][orbital]['coeff'])
    
    njl = np.array(fu.slater_coefficients[z][orbital]['n'])
    
    
    # for now we just state 1s contriutes to the core and 2s contributes to valence with a respective electron occupation of 2,1
  
    # we need array of R values to sample the electron density from so we can actually evaluate a fourier transofrm integral
    
    R_total = 0
   
    #Total radial finction is a superposition of these primitive radial functions and their corresponding expansion coefficent C_jln given 
    #in out hartree fock equation
    #delta is given next to each slater type orbital in the table 
    
    # after we fourier transform radial function to get form factor we need to use motte bethe formula to get to electron scattering factor
    # then compare with kirkland to check agreement and upscale
    
    
    for cj,zj,nj in zip(C,delta,njl):
        Nj = ((2*zj)**(nj+0.5))/(np.sqrt(math.factorial(2*nj)))
        S_j = Nj*r**(nj-1)*np.exp(-zj*r)          # each electron is defined by a primitive slater orbital of this form 
        R_total += cj*S_j
        
    
    return R_total      # now return the radial function for our atom use this function and integrate it to get form factor


def xray_form_factor_valence(r, rho, S, pv, k):
    """
    Vectorized valence form factor.
    r   : (Nr,) radial grid
    rho : (Nr,) valence density
    S   : (Ns,) array of momentum-transfer values
    pv  : number of valence electrons
    k   : kappa scaling factor
    """
    r2 = r**2                       # precompute r^2
    integrand_base = 4*np.pi *rho * r2  # shape (Nr,)

    # build matrix for all s at once
    sr = 2 * np.outer(S, r)         # shape (Ns, Nr)
    integrand = integrand_base[None, :] * np.sinc(sr)  # shape (Ns, Nr)

    # integrate along r for each s
    fQ = np.trapz(integrand, r, axis=1)

    return pv * fQ                   # scale by number of valence electrons


def xray_form_factor_core(r, rho, S, pc):
    """
    Vectorized core form factor.
    r   : (Nr,) radial grid
    rho : (Nr,) core density
    S   : (Ns,) array of momentum-transfer values
    pc  : number of core electrons
    """
    r2 = r**2
    integrand_base = 4*np.pi * rho * r2

    sr = 2 * np.outer(S, r)
    integrand = integrand_base[None, :] * np.sinc(sr)

    fQ = np.trapz(integrand, r, axis=1)

    return pc * fQ

    

def calc_scattering_amplitudes(q, Z ,pv,kappa):
    
   
    r_max = 20  # in angstrom
    n_points = 1000
    r = np.linspace(1e-6, r_max, n_points)
    
    core_orbitals = fu.elements_info[Z]['core_orbitals']
    valence_orbitals = fu.elements_info[Z]['valence_orbitals']
    core_density =0
    valence_density =0
    print(core_orbitals)
    print(valence_orbitals)
    n_e_core=0
    for orbital in core_orbitals:
       
        n_e_core += fu.elements_info[Z]['occupation'][orbital]
        R=  calc_slater_orbitals(Z,orbital,r)
       
        
        core_density += (R**2)
        
        
        
    core_density /= (4*np.pi)  
   
    
    core_density_n = core_density / np.trapz(4*np.pi*r**2*core_density, r)
    
    
    for orbital in valence_orbitals:
        
        R= (kappa**(3/2))*calc_slater_orbitals(Z,orbital,r*kappa)
        
       
        
        valence_density += (R**2)
        
        
        
    valence_density /= (4*np.pi)  
    valence_density_n = valence_density / np.trapz(4*np.pi*r**2*valence_density, r)   # need to normalize to 1 electron then scale by pv after 
  

    integral = 4 * np.pi * np.trapz(r**2 * valence_density, r)
    print("Integral of valence(r):", integral)
    
   # integral2 = 4 * np.pi * np.trapz(r**2 * core_density, r)
   # print("Integral of core(r):", integral2)
    
    

    
    #N_core =  (4*np.pi * np.trapz(r**2 * core_density, r))
   # N_valence = (4*np.pi * np.trapz(r**2 * valence_density, r))
  
   #core_density = (1/(4*np.pi))*(R_core**2)  # this will depend on n,l if multipolar is considered so its more complicate than this , just using this as an example 
   #valence_density = (1/(4*np.pi))*(R_valence**2)# wavefucniton = R(r)Y_l^m(theta,phi)
    pc = n_e_core
    
    
    
                       
    density_total = pc*core_density_n+ pv*valence_density_n#p_atom(r) in kappa formalism 
    Z_check = 4*np.pi * np.trapz(r**2 * density_total, r)
   # print("Total electrons:", Z_check)
    
   
    integrand = density_total*np.pi*r**2
    r2_expect = np.trapz(r**2*integrand, x=r)/np.trapz(integrand,x=r)   #mean square radius of electrons in the atom
    
    fu.elements_info[Z]["r2"] = r2_expect
    
    
    f_valence = xray_form_factor_valence(r,valence_density_n , q, pv, kappa) # so these actually work with g
    f_core = xray_form_factor_core(r, core_density_n, q, pc)  # works with g
   


    
    f_x_total =  f_core + f_valence    #fourier transform of the calculated radial funciton in 3d r from 0 to infinity 
    
                   
    return f_x_total 
   # return fe_core + fe_valence
    #return f_core + f_valence
   

def convert_x(Z, f_x, q):
    q = np.asarray(q)
    f_x = np.asarray(f_x)

    # Output array
    f_e = np.zeros_like(q, dtype=float)

    # Mask where q == 0
    mask0 = (q == 0)
    maskN = ~mask0  # q â‰  0

    # Special case for q = 0 (Ibers correction)
    r2 = fu.elements_info[Z]["r2"]
    f_e[mask0] = (Z * r2) / (3 * Bohr)

    # General Motteâ€“Bethe formula
    f_e[maskN] = (Z - f_x[maskN]) / (2 * np.pi**2 * Bohr * q[maskN]**2)   #* kappa**2 ?

    return f_e


# need to carefully look through and fix scaling of function 
#close but not quite


def kappa_factors(g_magnitude,Z,pv,kappa):
    
    S = g_magnitude / (2*np.pi)  
    
    
    return convert_x(Z,calc_scattering_amplitudes(S, Z, pv, kappa),S)
    #return calc_scattering_amplitudes(S, Z, pv, kappa)

#should handle values below 0.5 Q using kirkland values or some type of extrapolation

#Z = 3  # Li
Z = 41  # O
#Z = 41 # Nb
pv = 5  # 1 valence electron
kappa = 0.9
#pv of Nb is 5
Q = np.linspace(0, 10, 10000)  # momentum transfer array 1/bohr  so this is actually g_magnitude 


xsca = Q/(2*np.pi)

f_kappa = kappa_factors(Q, Z, pv, kappa).ravel()

f_kappa_1 = kappa_factors(Q, Z, pv, 1.1).ravel()

f_kappa_2 = kappa_factors(Q, Z, pv, 1).ravel()

f_kappa_3 = kappa_factors(Q, Z, pv, 1.3).ravel()

f_kirkland = px.f_kirkland(Z, Q).ravel() 

f_lobato = px.f_lobato(Z, Q).ravel() 

f_peng = px.f_peng(Z, Q).ravel() 

f_doyle_turner = px.f_doyle_turner(Z, Q).ravel() 
#we get correct values when passing S to both kirkland and kappa factors rather than Q or g magnitude

plt.plot(xsca, f_kappa, label='κ = 0.9')
plt.plot(xsca, f_kappa_2, label='κ = 1.0')
plt.plot(xsca, f_kappa_1, label='κ = 1.1')
plt.plot(xsca, f_kappa_3, label='κ = 1.3', color = "purple")
plt.plot(xsca, f_kirkland, label='Kirkland', linestyle='--')
#plt.plot(xsca, f_lobato, label='Lobato', linestyle='--')
#plt.plot(xsca, f_peng, label='Peng', linestyle='--')
#plt.plot(xsca, f_doyle_turner, label='Doyle turner', linestyle='--')



plt.xlabel('S (1/Å)')   # sin(theta)/lambda
plt.ylabel('Electron scattering factor f(Q)')
plt.legend()
plt.show()
end = time.time()

print(f"Total runtime of the program is {end - start} seconds")



