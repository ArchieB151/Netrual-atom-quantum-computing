# Load packages
import numpy as np
import matplotlib.pyplot as plt
from qutip import *

from basic_units import radians, degrees, cos

col = ['C0','C1','C2','C3']


# Function for phase computation
def phase_func(z):
    # a + ib
    a = np.real(z)
    b = np.imag(z)
    
    if b==0:
        ph = 2*np.pi
    if (a>=0 and b>0): # I
        ph = np.arctan(b/a) 
    if (a<0 and b>0): # II
        ph = np.arctan(b/a) + np.pi
    if (a<0 and b<0): # III
        ph = np.arctan(b/a) + np.pi
    if (a>0 and b<0): # IV
        ph = 2*np.pi + np.arctan(b/a)
        
    return ph

# Definition of the Hamiltonian for a two-qubit CZ gate
def hamiltonian(Omega,Delta):
    
    psi00 = tensor(basis(3,0),basis(3,0))
    psi01 = tensor(basis(3,1),basis(3,0)) 
    psi0r = tensor(basis(3,2),basis(3,0))
    psi10 = tensor(basis(3,0),basis(3,1))
    psi11 = tensor(basis(3,1),basis(3,1)) 
    psi1r = tensor(basis(3,2),basis(3,1))
    psir0 = tensor(basis(3,0),basis(3,2))
    psir1 = tensor(basis(3,1),basis(3,2))
    psirr = tensor(basis(3,2),basis(3,2))

    H0  = 0 * tensor(psi00.dag(),psi00)
    
    H01 = 1/2 * ( Omega * tensor(psi01.dag(),psi0r) + 
             np.conj(Omega) * tensor(psi0r.dag(),psi01) ) - Delta * tensor(psi0r.dag(),psi0r)
    
    H10 = 1/2 * ( Omega * tensor(psi10.dag(),psir0) + 
             np.conj(Omega) * tensor(psir0.dag(),psi10) ) - Delta * tensor(psir0.dag(),psir0)

    H2  = 1/2 * ( Omega * ( tensor(psi11.dag(),psir1) + tensor(psi11.dag(),psi1r) ) 
            + np.conj(Omega) * ( tensor(psir1.dag(),psi11) + tensor(psi1r.dag(),psi11) ) 
            ) - Delta/2 * ( tensor(psir1.dag(),psir1) + tensor(psir1.dag(),psi1r) 
                           + tensor(psi1r.dag(),psir1) + tensor(psi1r.dag(),psi1r))

    H = H0 + H01 + H10 + H2
    
    return H

# Optimal phase between two pulse
def exp_xi(Delta,Omega,tau):
    
    y = Delta/Omega
    s = Omega * tau
    
    a = np.sqrt(y**2+1)
    b = s*a/2
    
    return (a*np.cos(b) + 1j*y*np.sin(b)) / (-a*np.cos(b) + 1j*y*np.sin(b))

print('Theoretical xi:', phase_func(np.exp(-1j*3.90242)) )
print('xi calculated with the function:', phase_func(exp_xi(0.377371,1,4.29268))) 

H = Qobj( hamiltonian(1,0.377371),dims= [[3, 3, 3], [3, 3, 3]] )
fig, ax = matrix_histogram(H,limits=[-0.6,0.6])
ax.view_init(azim=-55, elev=10)
plt.show()

# Implementation of two-qubit CZ gate
def CZ_gate(psi,Omega,Delta,tau):
        
    # Times discretization
    times = np.linspace(0.0, tau, 200)
    
    # Apply first pulse
    H = hamiltonian(Omega,Delta)
    result = mesolve(H, psi, times,[], [])
    psi = result.states[-1]
    
    # Apply second pulse rotated by Omega -> Omega exp(i xi)
    H = hamiltonian(Omega * exp_xi(Delta,Omega,tau), Delta)
    result = mesolve(H, psi, times,[], [])
    psi = result.states[-1] 
        
    return psi

# Evolution of the system after the first pulse
def evol_CZ_gate(psi,Omega,Delta,tau,rho_ref):
        
    # Times discretization
    times = np.linspace(0.0, 3*tau, 200)
    
    # Apply first pulse
    H = hamiltonian(Omega,Delta)
    result = mesolve(H, psi, times,[], [rho_ref])
        
    return result

# Chain state initialization
def CZ_init(state_first,state_last):
    
    psi = basis(3,state_first) 
    psi = tensor(psi,basis(3,state_last))
    
    return psi

# Fix parameters
Omega   = 1
frac_DO = 0.377371
prod_Ot = 4.29268
Delta = frac_DO * Omega 
tau = prod_Ot / Omega

# Initialize state
state_first = 1
state_last = 1
psi_init = CZ_init(state_first,state_last) 

Delta_list = [0,frac_DO,4*frac_DO]
pop_b2_list = []
pop_a2_list = []
pop_b2_a2_list = []

for dd in Delta_list:
    
    rho_ref_b2 = ket2dm( np.sqrt(2)/2 * ( CZ_init(2,1) + CZ_init(1,2) ) )
    pop_b2 = evol_CZ_gate(psi_init,Omega,dd,tau,rho_ref_b2)   
    pop_b2_list.append(pop_b2)
    
    rho_ref_a2 = ket2dm( CZ_init(1,1) )
    pop_a2 = evol_CZ_gate(psi_init,Omega,dd,tau,rho_ref_a2)   
    pop_a2_list.append(pop_a2)    

    rho_ref_b2_a2 = rho_ref_b2 - rho_ref_a2
    pop_b2_a2 = evol_CZ_gate(psi_init,Omega,dd,tau,rho_ref_b2_a2 ) 
    pop_b2_a2_list.append(pop_b2_a2)
    
with plt.style.context('ggplot'):
    fig, ax = plt.subplots(3,1, figsize=(8,6))

    ax[0].plot(pop_b2_list[0].times, pop_b2_list[0].expect[0], color=col[0], ls='-', label=r'$\Delta/\Omega$=0')
    ax[0].plot(pop_b2_list[1].times, pop_b2_list[1].expect[0], color=col[1], ls='--',label=r'$\Delta/\Omega$=0.377')
    ax[0].plot(pop_b2_list[2].times, pop_b2_list[2].expect[0], color=col[2], ls='-.', label=r'$\Delta/\Omega$=$4\times$0.377')
    ax[0].set_ylabel(r'$\bf{P_{b_2}}$', fontsize=16)

    ax[0].set_ylim(-0.1,1.1)
    

    ax[1].plot(pop_a2_list[0].times, pop_a2_list[0].expect[0], color=col[0], ls='-',  label=r'$\Delta/\Omega$=0')
    ax[1].plot(pop_a2_list[1].times, pop_a2_list[1].expect[0], color=col[1], ls='--', label=r'$\Delta/\Omega$=0.377')
    ax[1].plot(pop_a2_list[2].times, pop_a2_list[2].expect[0], color=col[2], ls='-.', label=r'$\Delta/\Omega$=$4\times$0.377')
    ax[1].set_ylabel(r'$\bf{P_{a_2}}$', fontsize=16)

    ax[1].set_ylim(-0.1,1.1)
  

    ax[2].plot(pop_b2_a2_list[0].times, pop_b2_a2_list[0].expect[0], color=col[0], ls='-',  label=r'$\Delta/\Omega$=0')
    ax[2].plot(pop_b2_a2_list[1].times, pop_b2_a2_list[1].expect[0], color=col[1], ls='--', label=r'$\Delta/\Omega$=0.377')
    ax[2].plot(pop_b2_a2_list[2].times, pop_b2_a2_list[2].expect[0], color=col[2], ls='-.', label=r'$\Delta/\Omega$=$4\times$0.377')
    ax[2].set_ylabel(r'$\bf{P_{b_2}-P_{a_2}}$', fontsize=16)

    ax[2].set_xlabel(r'$\bf{Time}$ $[\Omega^{-1}]$', fontsize=16);
    ax[2].set_ylim(-1.1,1.1)
    

#ax[3].plot(pop_b2_a2_list[0].times, pop_b2_list[0].expect[0]-pop_a2_list[0].expect[0], label=r'$\Delta/\Omega$=0')
#ax[3].plot(pop_b2_a2_list[1].times, pop_b2_list[1].expect[0]-pop_a2_list[1].expect[0], label=r'$\Delta/\Omega$=0.377')
#ax[3].plot(pop_b2_a2_list[2].times, pop_b2_list[2].expect[0]-pop_a2_list[2].expect[0], label=r'$\Delta/\Omega$=$2\times$0.377')
#ax[3].grid(color='0.9')
#ax[3].set_ylim(-1.1,1.1)

# Put a legend above current axis


    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 3.85),fancybox=True, shadow=True, ncol=3)
  
    plt.show()