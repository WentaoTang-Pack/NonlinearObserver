import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
from random import sample
from scipy.integrate import odeint
from ChenFliess import *
from Reduction import *

def Osci(x, t, a=1.0, b=3.0):
    dx = np.zeros(2)
    dx[0] = a + x[0]**2 * x[1] - b*x[0] - x[0]
    dx[1] = b * x[0] - x[0]**2 * x[1]
    return dx

def OsciMeasurement(x):
    y = np.zeros(1)
    y[0] = x[0] + x[1]
    return y

def Lorentz(x, t, sigma=10.0, rho=28.0, beta=8.0/3.0, scale=10.0):
    dx = np.zeros(3)
    dx[0] = sigma*(x[1] - x[0])
    dx[1] = scale*x[0]*(rho/scale - x[2]) - x[1]
    dx[2] = scale*x[0]*x[1] - beta*x[2]
    return dx

def LorentzMeasurement(x):
    y = np.zeros(1)
    y[0] = x[1]
    return y


class Plant:
    def __init__(self, nx, ny, dynamics=Osci, measurement=OsciMeasurement):
        self.nx = nx
        self.ny = ny
        self.dynamics = dynamics
        self.measurement = measurement
        self.x = np.zeros(nx)
        self.y = self.measurement(self.x)
    def initialize(self, x0):
        self.x = x0
        self.y = self.measurement(self.x)
    def move_forward(self, t=0.0, dt=1.0):
        self.x += dt * self.dynamics(self.x, t)
        self.y = self.measurement(self.x)
    
    
class Observer:
    def __init__(self, nx, ny, maxlength=3, horizon=10):
        self.nx = nx
        self.ny = ny
        self.maxlength = maxlength
        self.horizon = horizon
        self.x = np.zeros(nx)
        self.xhistory = np.zeros((self.horizon + 1, self.nx))
        self.y = np.zeros(ny)
        self.yhistory = np.zeros((self.horizon + 1, self.ny))
        self.series = []
        
    def create(self):
        for i in range(self.nx):
            newseries = Series(self.maxlength, self.ny, self.horizon)
            self.series.append(newseries)
                
    def measure(self, plant):
        self.x = plant.x
        self.xhistory = np.delete(self.xhistory, 0, axis=0)
        self.xhistory = np.vstack([self.xhistory, self.x])
        self.y = plant.y
        self.yhistory = np.delete(self.yhistory, 0, axis=0)
        self.yhistory = np.vstack([self.yhistory, self.y])
    
    def measure_indirect(self, plant, kpca):
        self.x = kpca.principals
        self.xhistory = np.delete(self.xhistory, 0, axis=0)
        self.xhistory = np.vstack([self.xhistory, self.x])
        self.y = plant.y
        self.yhistory = np.delete(self.yhistory, 0, axis=0)
        self.yhistory = np.vstack([self.yhistory, self.y])
        
    def initialize(self, dt=1.0):
        for i in range(self.nx):
            obj = self.series[i].initialization_at_unity(self.xhistory[:,i], self.yhistory, dt)
    
    def update(self, rates, dt = 1.0):
        obj_old_vec = np.zeros(self.nx)
        obj_new_vec = np.zeros(self.nx)
        xobs_vec = np.zeros(self.nx)
        for i in range(self.nx):
            obj_old, obj_new = self.series[i].gradient_update(self.xhistory[:,i], self.yhistory, rates[i], dt)
            obj_old_vec[i] = obj_old
            obj_new_vec[i] = obj_new
            xobs = self.series[i].observe()
            xobs_vec[i] = xobs
        return obj_old_vec, obj_new_vec, xobs_vec
    
    
class Simulator:
    def __init__(self, model='Osci', observerrateconstant=1.0, kpcanu=0.5, kpcabeta=1.0, kpcahp=1.0):
        # Model information
        if model == 'Osci':
            self.nx = 2
            self.ny = 1
            self.dynamics = Osci
            self.measurement = OsciMeasurement
            self.dt = 0.02
            self.Nt = 2500
        if model == 'Lorentz':
            self.nx = 3
            self.ny = 1
            self.dynamics = Lorentz
            self.measurement = LorentzMeasurement
            self.dt = 0.01
            self.Nt = 5000
        # Observer hyperparameters
        self.observermaxlength = 3
        self.observerhorizon = 10
        self.observerwaittime = 0
        self.observerrates = self.nx * [observerrateconstant / self.dt]
        self.plant = Plant(nx=self.nx, ny=self.ny, dynamics=self.dynamics, measurement=self.measurement)
        self.observer = Observer(nx=self.nx, ny=self.ny, maxlength=self.observermaxlength, horizon=self.observerhorizon)
        self.observer.create()
        # Immersion
        self.immersiontaus = np.logspace(np.log10(0.5), np.log10(0.5*(2.0**self.nx)), num=self.nx+1)
        self.immersions = []
        self.nz = 0
        for i in range(self.ny):
            immersion = Immersion(self.immersiontaus, self.dt)
            self.immersions.append(immersion)
            self.nz += immersion.n
        # Dimensionality reduction hyperparameters
        self.kpcanu = kpcanu
        self.kpcabeta = kpcabeta
        self.kpcahp = kpcahp
        self.KPCA = KPCA(n=self.nx+1, r=self.nx, nu=self.kpcanu, eta=self.kpcabeta*self.dt, kernel=kernel_Gaussian, kernelhp=self.kpcahp)
        # Records
        self.Nt += self.observerhorizon + self.observerwaittime - 1
        self.xtrajectory = np.zeros((self.Nt + 1, self.nx))
        self.ytrajectory = np.zeros((self.Nt + 1, self.ny))
        self.ztrajectory = np.zeros((self.Nt + 1, self.nz))
        self.principalstrajectory = np.zeros((self.Nt + 1, self.nx))
        self.principalsposteriortrajectory = np.zeros((self.Nt + 1, self.nx))
        self.xobstrajectory = np.zeros((self.Nt + 1, self.nx))
        self.Joldtrajectory = np.zeros((self.Nt + 1, self.nx))
        self.Jnewtrajectory = np.zeros((self.Nt + 1, self.nx))
        
    def simulate(self, x0 = np.array([1.0, 1.0]), xinfo=True):
        x0 = x0
        if (len(x0) != self.nx):
            x0 = np.array(self.nx * [x0[0]])
        self.plant.initialize(x0)
        self.xtrajectory[0, :] = self.plant.x
        self.ytrajectory[0, :] = self.plant.y
        self.observer.initialize(dt=self.dt)
        J = 0.
        if xinfo:
            self.observerwaittime = 0
        else:
            self.observerwaittime = 500

        for tcount in range(1, self.Nt + 1):
            # Time moves forward
            self.plant.move_forward(t=tcount*self.dt, dt=self.dt)
            for i in range(self.ny):
                self.immersions[i].move_forward(self.plant.y[i])
            self.xtrajectory[tcount, :] = self.plant.x
            self.ytrajectory[tcount, :] = self.plant.y
            zs = np.concatenate([immersion.z for immersion in self.immersions])
            self.ztrajectory[tcount, :] = zs
            if tcount >= (self.observerhorizon + self.observerwaittime):
                # Reduce dimensionality
                if (tcount == self.observerhorizon + self.observerwaittime):
                    if xinfo:
                        self.KPCA.initialize(x0=zs)
                    else:
                        # random points for initializing KPCA
                        pointindex = sample(range(tcount - int(self.observerwaittime/2), tcount + 1), self.KPCA.r)
                        points = self.ztrajectory[pointindex, :]
                        self.KPCA.initialize_multiple(points) #self.KPCA.initialize(x0=zs)
                self.KPCA.execute(zs, (tcount - 1)*self.dt)
                self.principalstrajectory[tcount, :] = self.KPCA.principals
                # Perform measurements
                if xinfo:
                    self.observer.measure(self.plant)
                else:
                    self.observer.measure_indirect(self.plant, self.KPCA)
                # Observer update
                obj_old_vec, obj_new_vec, xobs_vec = self.observer.update(rates=self.observerrates, dt=self.dt)
                self.Joldtrajectory[tcount, :] = obj_old_vec
                self.Jnewtrajectory[tcount, :] = obj_new_vec
                self.xobstrajectory[tcount, :] = xobs_vec
            if (tcount % 100 == 0):
                print('Sampling time', tcount)
        
        # CALCULATE THE PERFORMANCE
        if xinfo:
            for tcount in range(self.observerhorizon + self.observerwaittime, self.Nt + 1):
                J += np.linalg.norm(self.xobstrajectory[tcount, :] - self.xtrajectory[tcount, :], 2)**2 
        else:
            for tcount in range(self.observerhorizon + self.observerwaittime, self.Nt + 1):
                p = self.KPCA.posterior_calculate_PC(self.ztrajectory[tcount, :])
                self.principalsposteriortrajectory[tcount, :] = p
                J += np.linalg.norm(self.xobstrajectory[tcount, :] - p, 2)**2 
        J = np.sqrt(J/(self.Nt + 1 - self.observerhorizon - self.observerwaittime))
        return J
        
        
    def plot(self, xinfo=True):
        trange = np.arange(self.Nt + 1 - self.observerhorizon - self.observerwaittime) * self.dt
        
        # Plot the x components separately
        fig, ax = plt.subplots()
        for i in range(self.nx):
            ax = plt.subplot(self.nx, 1, i + 1)
            if xinfo:
                ax.plot(trange, self.xtrajectory[self.observerhorizon + self.observerwaittime:, i], 'b', label=r'$x_{%s}$' % (i+1))
                ax.plot(trange, self.xobstrajectory[self.observerhorizon + self.observerwaittime:, i], 'r', label=r'$\hat{x}_{%s}$' % (i+1))
            else:
                ax.plot(trange, self.principalsposteriortrajectory[self.observerhorizon + self.observerwaittime:, i], 'k', label=r'$\pi_{%s}^\ast$' % (i+1))
                ax.plot(trange, self.principalstrajectory[self.observerhorizon + self.observerwaittime:, i], 'b', label=r'$\pi_{%s}$' % (i+1))
                ax.plot(trange, self.xobstrajectory[self.observerhorizon + self.observerwaittime:, i], 'r', label=r'$\hat{\pi}_{%s}$' % (i+1))
            ax.legend()
            if i == self.nx - 1:
                ax.set_xlabel(r'$t$')
        figurename = '%s_xinfo_%s_alpha_%d_nu_%d_beta_%d_sigma_%d_xplot.eps' % (self.dynamics.__name__, xinfo, int(self.observerrates[0]*self.dt*100), int(self.kpcanu*100), int(self.kpcabeta), int(self.kpcahp*100))
        fig.savefig(figurename, format='eps', dpi=1200)
        plt.show()
        # Save x components to csv file
        if xinfo:
            csvdata = np.hstack([trange.reshape((-1,1)), self.xtrajectory[self.observerhorizon + self.observerwaittime:, :], self.xobstrajectory[self.observerhorizon + self.observerwaittime:, :]])
            header = 't' + ',x'*self.nx + ',xhat'*self.nx
        else:
            csvdata = np.hstack([trange.reshape((-1,1)), self.principalsposteriortrajectory[self.observerhorizon + self.observerwaittime:, :], self.principalstrajectory[self.observerhorizon + self.observerwaittime:, :], self.xobstrajectory[self.observerhorizon + self.observerwaittime:, :]])
            header = 't' + ',pistar'*self.nx + ',pi'*self.nx + ',pihat'*self.nx
        csvname = '%s_xinfo_%s_alpha_%d_nu_%d_beta_%d_sigma_%d_xplot.csv' % (self.dynamics.__name__, xinfo, int(self.observerrates[0]*self.dt*100), int(self.kpcanu*100), int(self.kpcabeta), int(self.kpcahp*100))
        np.savetxt(csvname, csvdata, delimiter=",", header=header)

        
        # Plot the z components separately
        fig, ax = plt.subplots()
        if not xinfo:
            for i in range(self.nz):
                ax = plt.subplot(self.nz, 1, i + 1)
                ax.plot(trange, self.ztrajectory[self.observerhorizon + self.observerwaittime:, i], 'b', label=r'$z_{%s}$' % (i+1))
                ax.legend()
            if i == self.nz - 1:
                ax.set_xlabel(r'$t$')
            figurename = '%s_xinfo_%s_alpha_%d_nu_%d_beta_%d_sigma_%d_zplot.eps' % (self.dynamics.__name__, xinfo,int(self.observerrates[0]*self.dt*100), int(self.kpcanu*100),int(self.kpcabeta), int(self.kpcahp*100))
            fig.savefig(figurename, format='eps', dpi=1200)
            plt.show()
        
        # Plot the x components in a phase portrait
        if ((not xinfo) and (self.nx == 2)):
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(self.xtrajectory[self.observerhorizon + self.observerwaittime:, 0], self.xtrajectory[self.observerhorizon + self.observerwaittime:, 1], color='k')
            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')
            figurename = '%s_xinfo_%s_alpha_%d_nu_%d_beta_%d_sigma_%d_xphase.eps' % (self.dynamics.__name__, xinfo,int(self.observerrates[0]*self.dt*100), int(self.kpcanu*100),int(self.kpcabeta), int(self.kpcahp*100))
            fig.savefig(figurename, format='eps', dpi=1200)
            plt.show()
        
        # Plot the z components in a phase portrait
        if ((not xinfo) and (self.nz == 3)):
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot(self.ztrajectory[self.observerhorizon + self.observerwaittime:, 0], self.ztrajectory[self.observerhorizon + self.observerwaittime:, 1], self.ztrajectory[self.observerhorizon + self.observerwaittime:, 2],color='b')
            ax.set_xlabel(r'$z_1$')
            ax.set_ylabel(r'$z_2$')
            ax.set_zlabel(r'$z_3$')
            figurename = '%s_xinfo_%s_alpha_%d_nu_%d_beta_%d_sigma_%d_zphase.eps' % (self.dynamics.__name__, xinfo,int(self.observerrates[0]*self.dt*100), int(self.kpcanu*100),int(self.kpcabeta), int(self.kpcahp*100))
            fig.savefig(figurename, format='eps', dpi=1200)
            plt.show()
        
        # Plot the principal components in a phase portrait
        if ((not xinfo) and (self.nx == 2)):
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.plot(self.principalstrajectory[self.observerhorizon + self.observerwaittime:, 0], self.principalstrajectory[self.observerhorizon + self.observerwaittime:, 1], color='m')
            ax.set_xlabel(r'$\pi_1$')
            ax.set_ylabel(r'$\pi_2$')
            figurename = '%s_xinfo_%s_alpha_%d_nu_%d_beta_%d_sigma_%d_pcphase.eps' % (self.dynamics.__name__, xinfo,int(self.observerrates[0]*self.dt*100), int(self.kpcanu*100),int(self.kpcabeta), int(self.kpcahp*100))
            fig.savefig(figurename, format='eps', dpi=1200)
            plt.show()
        
        # Save phase portraits to csv file
        if not xinfo:
            csvdata = np.hstack([trange.reshape((-1,1)), self.xtrajectory[self.observerhorizon + self.observerwaittime:, :], self.ztrajectory[self.observerhorizon + self.observerwaittime:, :], self.principalstrajectory[self.observerhorizon + self.observerwaittime:, :]])
            header = 't' + ',x'*self.nx + ',z'*self.nz + ',pi'*self.nx
            csvname = '%s_xinfo_%s_alpha_%d_nu_%d_beta_%d_sigma_%d_phaseportraits.csv' % (self.dynamics.__name__, xinfo, int(self.observerrates[0]*self.dt*100), int(self.kpcanu*100), int(self.kpcabeta), int(self.kpcahp*100))
            np.savetxt(csvname, csvdata, delimiter=",", header=header)
        
        
        # Plot the KPCA dataset
        if not xinfo:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            if self.KPCA.X.shape[1] <= 3:
                ax.scatter(self.KPCA.X[:,0], self.KPCA.X[:,1], self.KPCA.X[:,2], marker='o')
            else:
                ax.scatter(self.KPCA.X[:,0], self.KPCA.X[:,1], self.KPCA.X[:,2], c=self.KPCA.X[:,3], marker='o', cmap='jet')
            ax.set_xlabel(r'$z_1$')
            ax.set_ylabel(r'$z_2$')
            ax.set_zlabel(r'$z_3$')
            figurename = '%s_xinfo_%s_alpha_%d_nu_%d_beta_%d_sigma_%d_PCApoints.eps' % (self.dynamics.__name__, xinfo,int(self.observerrates[0]*self.dt*100), int(self.kpcanu*100),int(self.kpcabeta), int(self.kpcahp*100))
            fig.savefig(figurename, format='eps', dpi=1200)
            plt.show()
        if not xinfo:
            csvdata = self.KPCA.X
            csvname = '%s_xinfo_%s_alpha_%d_nu_%d_beta_%d_sigma_%d_PCApoints.csv' % (self.dynamics.__name__, xinfo,int(self.observerrates[0]*self.dt*100), int(self.kpcanu*100),int(self.kpcabeta), int(self.kpcahp*100))
            np.savetxt(csvname, csvdata, delimiter=",")