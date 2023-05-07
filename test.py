import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from scipy.integrate import odeint
from Dynamics import * 
from Reduction import *
    
# Standard tuning for Oscillator: model='Osci', observerrateconstant=1.0, kpcanu=0.05, kpcabeta=1.0, kpcahp=0.5
# Standard tuning for Lorentz: model='Lorentz', observerrateconstant=5.0, kpcanu=0.05, kpcabeta=1.0, kpcahp=0.5
xinfo = True
sim = Simulator(model='Osci', observerrateconstant=1.0, kpcanu=0.05, kpcabeta=1.0, kpcahp=0.5)
J = sim.simulate(x0 = np.array([1.0, 1.0]), xinfo=xinfo)
print('Averaged observation error =', J)
sim.plot(xinfo=xinfo)
# Understanding the principal components
# if not xinfo:
#     norms = np.array([np.linalg.norm(sim.KPCA.alpha[:,i]) for i in range(sim.KPCA.r)])
#     print('alpha norms:', norms)
#     angles = np.zeros((sim.KPCA.r, sim.KPCA.r))
#     for i in range(sim.KPCA.r):
#         for j in range(sim.KPCA.r):
#             angles[i, j] = np.arccos(min(1.0, max(-1.0, np.dot(sim.KPCA.alpha[:, i], sim.KPCA.alpha[:, j]) / norms[i] / norms[j])))
#     print('alpha angles:', angles * 180.0 / np.pi)