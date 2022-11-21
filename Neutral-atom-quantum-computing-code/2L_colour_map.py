import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import kron, eig


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
ns = 1
Gamma_temp = 0.1
Omega_temp = 1
Gamma=2*np.pi*Gamma_temp #input parameter
Omega=2*np.pi*Omega_temp

phiL=  0
Delta_master = []
max_transfer = []

L=np.zeros((4,4))
L[0,3]=Gamma
L[1,1]=-Gamma/2
L[2,2]=-Gamma/2
L[3,3]=-Gamma

npts=500
tmax=10.0
end = 10
dem = 100
time_step = np.pi/Omega
res = time_step/dem
n = end/res
npts = round(n)
t= np.arange(0,end,res)      

master_rho_bb =[]
for i in range(-200,200):
    print(i)
    Delta = i/10
    Delta_master.append(Delta)
    H=np.array([[Delta/2, (Omega/2)],[(Omega/2), -Delta/2] ])
    I2=np.eye(2,2)

    Hrho=kron(H,I2)
    rhoH=kron(I2,np.conj(H))

    evals, evecs = eig(-1.j*(Hrho-rhoH)+L)
    evecs=np.mat(evecs)

 
   

    rho0=np.zeros((4,1))
    rho0[0]=1.0
              
    rho_bb=np.zeros(npts)
    rho_gg = np.zeros(npts)

    for i in range(0,npts):
        rho=evecs*np.mat(np.diag(np.exp(evals*t[i])))*np.linalg.inv(evecs)*rho0
        rho_bb[i]=rho[3].real
        rho_gg[i]=rho[0].real

    master_rho_bb.append(list(rho_bb[0:400]))

   
    


print(master_rho_bb)
print(np.size(master_rho_bb))
with plt.style.context('ggplot'):
    plt.imshow(master_rho_bb, cmap='plasma', interpolation='nearest', extent=[0,40,-20,20])
    plt.xlabel(r'time, $10^{-7} s$')
    plt.ylabel(r'Detuning $\Delta /2 \pi$')
    plt.colorbar(label='Probability of being in the excited state')
    plt.title(r'Colour map of two level detuning')
    plt.tight_layout()
    plt.grid(False)
    plt.show()

