import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import kron, eig


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
ns = 1
Gamma_e_temp = 1
Gamma_r_temp = 1
Omega_i_temp = 1
Omega_e_temp = 1
delta = 1000

Gamma_e=2*np.pi*Gamma_e_temp #input parameter
Gamma_r=2*np.pi*Gamma_r_temp
Omega_i=2*np.pi*Omega_i_temp
Omega_e = 2*np.pi*Omega_e_temp


rho0=np.zeros((9,1))
rho0[0]=1.0
npts=500
tmax=10.0
end = 10
dem = 100
rabi_freq = Omega_i
time_step = np.pi/rabi_freq
res = time_step/dem
n = end/res
npts = round(n)
t= np.arange(0,end,res)                
rho_bb=np.zeros(npts)
rho_gg = np.zeros(npts)
Delta_master = []
max_transfer = []


L=np.zeros((9,9))
L[0,4]= Gamma_e 
L[1,1]=-Gamma_e/2
L[2,2]=-Gamma_r/2
L[3,3]=-Gamma_e/2
L[4,4]=-Gamma_e 
L[4,8] = Gamma_r
L[5,5]=-(Gamma_e+Gamma_r)/2
L[6,6] =-Gamma_r/2
L[7,7]=-(Gamma_e+Gamma_r)/2
L[8,8]=-Gamma_r

storage = []
rho_bb = []
rho_gg = []
for i in range(-200,200):
    print(i)
    
    Delta = i/5
    Delta_master.append(Delta)
    H=np.array([[0, Omega_i/2,0 ],[Omega_i/2,-2*Delta,Omega_e/2],[0,Omega_e/2,-delta]])
    I2=np.eye(3,3)

    Hrho=kron(H,I2)
    rhoH=kron(I2,np.conj(H))

    

    evals, evecs = eig(-1.j*(Hrho-rhoH)+L)
    evecs=np.mat(evecs)
    storage.append(evecs)
    rho=evecs*np.mat(np.diag(np.exp(evals*t[dem])))*np.linalg.inv(evecs)*rho0


    rho_4 = rho[4].real
    rho_0 = rho[0].real
    rho_0 = float(rho_0[0][0])
    rho_4 = float(rho_4[0][0])

    rho_bb.append(rho_4)
    rho_gg.append(rho_0)
    print(rho_0)
    
    
   

with plt.style.context('ggplot'):
    plt.plot(Delta_master, rho_bb)
    plt.title(r'Detuning $\Delta$, $\Omega_e/2\pi =$' +str(round(Omega_e_temp,5)) + ','+ r'$\Gamma_e/2\pi =$ ' + str(round(Gamma_e_temp,5)) +','+ r'$\Omega_i/2\pi =$' +str(round(Omega_i_temp,5)) + ','+ r'$\Gamma_r/2\pi =$ ' + str(round(Gamma_r_temp,5)) + ','+ r'$\delta/2\pi =$ ' + str(round(delta,5)))
    plt.xlabel(r'Detuning $\Delta / 2\pi$')
    plt.ylabel(r'Probability to be in excited state $P_e$')
    plt.ylim([-0.05, 1.05])
    plt.show()

    
   




