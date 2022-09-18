import numpy as np
from qutip import ket2dm, Bloch
def bloch_vector(state):
    c0=state[0,0]
    c1=state[1,0]
    if c0==0:
        px=0
        py=0
        pz=-1
    else:
        u=c1/c0
        px=2*u.real/(1+u.real**2+u.imag**2)
        py = 2*u.imag/(1+u.real**2+u.imag**2)
        pz = (1-u.real**2-u.imag**2)/(1+u.real**2+u.imag**2) 
    return [px,py,pz]
def toBloch(matrix):
    [[a, b], [c, d]] = matrix
    x = complex(c + b).real
    y = complex(c - b).imag
    z = complex(a - d).real
    return x, y, z
def to_bloch_qobj(dm):
    matrix = np.array(dm.data.todense()[:2,:2])
    [[a, b], [c, d]] = matrix
    x = complex(c + b).real
    y = complex(c - b).imag
    z = complex(a - d).real
    return x, y, z
def show_bloch(event, tlist, states, subsystem, normalize=False, az_el=None):
    x=event.xdata
    y=event.ydata
    delta = tlist[1]-tlist[0]
    i = int(np.round((x-tlist[0])/delta))
    rho = states[i]
    if not rho.isoper:
        rho = ket2dm(rho)
    if subsystem is not None:
        rho = rho.ptrace(subsystem)
    rho /= rho.tr()
    [x,y,z] = to_bloch_qobj(rho)
    r, theta, phi = to_bloch_polar(rho)
    if normalize:
        x /= r
        y /= r
        z /= r
    b = Bloch()
    if az_el is not None:
        b.view= az_el
    b.font_size=15
    b.add_points([x,y,z])
    b.add_annotation([0,0,-1.4], \
        f'r={np.round_(r,1)} t={np.round_(np.degrees(theta))} p={np.round_(np.degrees(phi))} {np.round_(x,1)} {np.round_(y,2)} {np.round_(z,1)}')
    b.show()

def bloch_trajectory(b,states, subsystem, normalize=False, title=None, az_el=None, fig=None):
    for i in range(len(states)):
        rho = states[i]
        if not rho.isoper:
            rho = ket2dm(rho)
        if subsystem is not None:
            rho = rho.ptrace(subsystem)
        rho /= rho.tr()
        [x,y,z] = to_bloch_qobj(rho)
        r, theta, phi = to_bloch_polar(rho)
        if normalize:
            x /= r
            y /= r
            z /= r
        b.font_size=15
        b.add_points([x,y,z])
    if title is not None:
        b.add_annotation([0,0,-1],title)
    if az_el is not None:
        b.view = az_el

def to_bloch_polar(dm):
    matrix = np.array(dm.data.todense()[:2,:2])
    [[a, b], [c, d]] = matrix
    x = complex(c + b).real
    y = complex(c - b).imag
    z = complex(a - d).real
    
    phi = np.arctan2(y,x)
    while (phi<0):
        phi += 2*np.pi
    theta = np.arctan2(np.sqrt(x**2+y**2),z)
    if theta <0:
        theta = -theta
    r= np.sqrt(x**2+y**2+z**2)
    return r, theta, phi

def bloch_angles(state):
    c0=state[0]
    c1=state[1]
    if c0==0:
        theta=0
        phi=0
    else:
        phi=np.angle(c0)
        c1 *= np.exp(-1j*phi)
        phi = np.angle(c1)
        while phi<0:
            phi += 2*np.pi
        theta = 2*np.arctan2(np.abs(c1),np.abs(c0))
    return theta, phi
def bloch_angles_qobj(state):
    state = state.data.todense()
    c0=state[0]
    c1=state[1]
    if c0==0:
        theta=0
        phi=0
    else:
        phi=np.angle(c0)
        c1 *= np.exp(-1j*phi)
        phi = np.angle(c1)
        while phi<0:
            phi += 2*np.pi
        theta = 2*np.arctan2(np.abs(c1),np.abs(c0))
    return theta, phi