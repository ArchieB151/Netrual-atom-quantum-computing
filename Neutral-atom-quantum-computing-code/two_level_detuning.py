import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import kron, eig


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

Gamma=0.1 #input parameter
Omega=2

phiL=  0#1.0*np.pi/2
Delta_master = []
max_transfer = []
for i in range(-100,100):
    print(i)
    Delta = i/10
    Delta_master.append(Delta)
    H=np.array([[Delta/2, (Omega/2)*np.exp(-1.j*phiL)],[(Omega/2)*np.exp(1.j*phiL), -Delta/2] ])
    I2=np.eye(2,2)

    Hrho=kron(H,I2)
    rhoH=kron(I2,np.conj(H))

    L=np.zeros((4,4))
    L[0,3]=Gamma
    L[1,1]=-Gamma/2
    L[2,2]=-Gamma/2
    L[3,3]=-Gamma

    evals, evecs = eig(-1.j*(Hrho-rhoH)+L)
    evecs=np.mat(evecs)

    rabi_freq = Omega
    time_step = np.pi/rabi_freq

    rho0=np.zeros((4,1))
    rho0[0]=1.0
    npts=500
    tmax=10.0
    end = 10
    dem = 100
    res = time_step/dem
    n = end/res
    npts = round(n)
    t= np.arange(0,end,res)                #np.linspace(0,tmax*time_step,npts)
    rho_bb=np.zeros(npts)
    rho_gg = np.zeros(npts)

    for i in range(0,npts):
        rho=evecs*np.mat(np.diag(np.exp(evals*t[i])))*np.linalg.inv(evecs)*rho0
        rho_bb[i]=rho[3].real
        rho_gg[i]=rho[0].real

    test = find_nearest(t,time_step)

    temp3 = rho_bb[5*dem]
    max_transfer.append(temp3)



with plt.style.context('ggplot'):
    plt.plot(Delta_master, max_transfer)
 
    plt.show()
