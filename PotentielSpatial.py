import numpy as np
#import numpy.fft as fft
from numpy.fft import ifft2
import scipy as sp
from scipy.integrate import odeint,solve_ivp,RK45
import scipy.special as spc
from scipy import interpolate
import matplotlib.pyplot as plt
from tqdm import tqdm

class PotentielSpatial:
    """
	Hamilton class gathering the potential calculation and particles dynamics
    Inputs are:
        - B magnetic field
        - e sign charge
        - eta 
        - w0
        - k
        - phi1
    """
    def __init__(self,eta,A,rho,nbp,M,noeudx,noeudy,T,dt,choixInt,choixOrdreGyro,choixOrdreFLR,ntheta):
        # PARSING INPUTS 
        #self.B = B 			# champ magnétique B
        self.A = A
        self.Ab = A*spc.jv(0,rho*np.sqrt(2))		# amplitude Phi modifier
        self.rho = rho 		# rayon Larmor
        self.nbp = nbp		# nombres particules
        self.M = M 			# nombre de mode champ electrostatique
        #self.x = 0 		# liste position particule dans le temps en x
        self.x = np.zeros(nbp)
        self.x[:] = np.random.rand(nbp)*2*np.pi
        #self.y = 0		# liste position particule dans le temps en y
        self.y = np.zeros(nbp)
        self.y[:] = np.random.rand(nbp)*2*np.pi
        self.T = T 				# Temps final
        self.dt = dt			# Nombre iteration dans le temps
        self.noeudx = int(noeudx)		# taille maille x
        self.noeudy = int(noeudy)		# taille maille y 
        self.choixInt = choixInt
        self.choixOrdreGyro = choixOrdreGyro
        self.eta = eta
        if rho != 0:
            self.theta = np.linspace(0,2*np.pi,ntheta)
            self.rhoc = self.rho*np.cos(self.theta)
            self.rhos = self.rho*np.sin(self.theta)

        if choixOrdreFLR == 2:
        	self.Ab = A*(1-rho**2/2)

        if rho == 0:
        	self.choixOrdreGyro = 0

    def RungeKutta(self,model,u0,Tf,dt):
        u = np.zeros((2*self.nbp,int(Tf/dt)+1))
        u[:,0] = u0
        print(u0)
        for i in tqdm(range(int(Tf/dt))):
            
            du0 = model(dt*i,u[:,i])
            du1 = model(dt*(i+1/2),u[:,i]+dt*du0/2)
            du2 = model(dt*(i+1/2),u[:,i]+dt*du1/2)
            du3 = model(dt*(i+1),u[:,i]+dt*du2)
            u[:,i+1] = u[:,i] + dt*(du0+2*du1+2*du2+du3)/6
        return u

    def carte(self):
        x = np.arange(0,self.noeudx)*2*np.pi/self.noeudx
        y = np.arange(0,self.noeudy)*2*np.pi/self.noeudy
        M=self.M
        A=self.A
        phases = 2.0 * np.pi * np.random.random((M, M))
        n = np.meshgrid(np.arange(1, M+1), np.arange(1, M+1))

        fft_phi = np.zeros((self.noeudx, self.noeudy), dtype=np.complex128)
        fft_dphidx = np.zeros((self.noeudx, self.noeudy), dtype=np.complex128)
        fft_dphidy = np.zeros((self.noeudx, self.noeudy), dtype=np.complex128)

        fft_phi[1:M+1, 1:M+1] = A * (1.0 / (n[0] ** 2 + n[1] ** 2) ** 1.5).astype(np.complex128) * np.exp(1j * phases)
        fft_dphidx[1:M+1, 1:M+1] = A *1j* (n[0] / (n[0] ** 2 + n[1] ** 2) ** 1.5).astype(np.complex128) * np.exp(1j * phases)
        fft_dphidy[1:M+1, 1:M+1] = A *1j* (n[1] / (n[0] ** 2 + n[1] ** 2) ** 1.5).astype(np.complex128) * np.exp(1j * phases) 

        phi = ifft2(fft_phi) * self.noeudx*self.noeudy
        dphidx = ifft2(fft_dphidx) * self.noeudx*self.noeudy
        dphidy = ifft2(fft_dphidy) * self.noeudx*self.noeudy

        self.dphi1dx = interpolate.interp2d(x, y, np.real(dphidx))
        self.dphi2dx = interpolate.interp2d(x, y, np.imag(dphidx))
        self.dphi1dy = interpolate.interp2d(x, y, np.real(dphidy))
        self.dphi2dy = interpolate.interp2d(x, y, np.imag(dphidy))
        #self.dphidx = interpolate.interpn(x, y, dphidx)
        #self.dphidy = interpolate.interpn(x, y, dphidy)

        """
        phasebb = np.random.rand(self.M,self.M)*2*np.pi
        phaseb = np.zeros((self.noeudy,self.noeudx,self.M,self.M))
        phaseb[:,:] = phasebb
        phase = phaseb.transpose(0,2,1,3)

        n = np.arange(1,self.M+1)
        nn,m = np.meshgrid(n,n)
        nb,xx = np.meshgrid(n,x)
        nb,yy = np.meshgrid(n,y)

        nx = np.zeros((y.shape[0],self.M,x.shape[0],self.M))
        myb = np.zeros((x.shape[0],self.M,y.shape[0],self.M))
        myb[:,:] = n*yy
        nx[:,:] = n*xx
        my = myb.transpose(2,1,0,3)
        sinus = np.sin(nx+my+phase).transpose(0,2,1,3)
        cosinus = np.cos(nx+my+phase).transpose(0,2,1,3)

        z1x = self.A*np.sum(np.sum(nn*cosinus/((nn**2+m**2)**(3/2)),axis=2),axis=2)
        z2x = -self.A*np.sum(np.sum(nn*sinus/((nn**2+m**2)**(3/2)),axis=2),axis=2)
        z1y = self.A*np.sum(np.sum(m*cosinus/((nn**2+m**2)**(3/2)),axis=2),axis=2)
        z2y = -self.A*np.sum(np.sum(m*sinus/((nn**2+m**2)**(3/2)),axis=2),axis=2)

        self.dphi1dx = interpolate.interp2d(x,y,z1x)
        self.dphi2dx = interpolate.interp2d(x,y,z2x)
        self.dphi1dy = interpolate.interp2d(x,y,z1y)
        self.dphi2dy = interpolate.interp2d(x,y,z2y)
        """

    def Potential0(self,x,y,t):
        dphidy = self.dphi2dy(x%(2*np.pi),y%(2*np.pi))*np.cos(t)-self.dphi1dy(x%(2*np.pi),y%(2*np.pi))*np.sin(t)
        dphidx = self.dphi2dx(x%(2*np.pi),y%(2*np.pi))*np.cos(t)-self.dphi1dx(x%(2*np.pi),y%(2*np.pi))*np.sin(t)

        return dphidy,dphidx

    def Potential1(self,x,y,t):
        xb,rhocb = np.meshgrid(x,self.rhoc)
        xrho = np.reshape(xb + rhocb,(xb.shape[0]*xb.shape[1]),order='F')
        yb,rhosb = np.meshgrid(y,self.rhos)
        yrho = np.reshape(yb - rhosb,(yb.shape[0]*yb.shape[1]))

        dphidy = np.mean(np.reshape(self.dphi1dy(xrho,yrho).diagonal()*np.cos(t)-self.dphi2dy(xrho,yrho).diagonal()*np.sin(t),(xb.shape[0],xb.shape[1]),order='F'),axis=0)
        dphidx = np.mean(np.reshape(self.dphi1dx(xrho,yrho).diagonal()*np.cos(t)-self.dphi2dx(xrho,yrho).diagonal()*np.sin(t),(xb.shape[0],xb.shape[1]),order='F'),axis=0)

        return dphidy,dphidx

    def Potential2(self,x,y,t):
        dphidy = self.dphi1dy(x%(2*np.pi),y%(2*np.pi))*np.cos(t)-self.dphi2dy(x%(2*np.pi),y%(2*np.pi))*np.sin(t)
        dphidx = self.dphi1dx(x%(2*np.pi),y%(2*np.pi))*np.cos(t)-self.dphi2dx(x%(2*np.pi),y%(2*np.pi))*np.sin(t)

        return dphidy,dphidx

    def Model0(self,t,X):
        x = X[:self.nbp]
        y = X[self.nbp:]
        dXdt = np.zeros(2*self.nbp)

        dXdtb = self.Potential0(x,y,t)
        dXdt[:self.nbp] = -dXdtb[0].diagonal()
        dXdt[self.nbp:] = dXdtb[1].diagonal()

        return dXdt

    def Model1(self,t,X):
        x = X[:self.nbp]
        y = X[self.nbp:]
        dXdt = np.zeros(2*self.nbp)

        dXdtb = self.Potential1(x,y,t)
        dXdt[:self.nbp] = -dXdtb[0]
        dXdt[self.nbp:] = dXdtb[1]

        return dXdt

    def Model2(self,t,X):
        x = X[:self.nbp]
        y = X[self.nbp:]
        dXdt = np.zeros(2*self.nbp)

        dXdtb = self.Potential0(x,y,t)
        dXdt[:self.nbp] = -dXdtb[0].diagonal()
        dXdt[self.nbp:] = dXdtb[1].diagonal()

        return dXdt

    def Dynamics(self):

        x = self.x
        y = self.y

        self.carte()

        X0 = np.zeros(2*self.nbp)
        X0[:self.nbp] = x[:]
        X0[self.nbp:] = y[:]

        if self.choixOrdreGyro == 0:
            Modelx = self.Model0
        elif self.choixOrdreGyro == 1:
            Modelx = self.Model1
        else:
            Modelx = self.Model2

        if self.choixInt==0:
            t = 2*np.pi*np.arange(self.T)

            s = odeint(Modelx,X0,t,atol=10**(-6),rtol=10**(-6),tfirst=True)

            x = s[:,:self.nbp]
            y = s[:,self.nbp:]

        elif self.choixInt==1:
            t = 2*np.pi*np.arange(int(self.T))

            s = solve_ivp(Modelx,(0,2*np.pi*int(self.T)),X0,t_eval=t,atol=10**(-6),rtol=10**(-6))

            x = s.y[:self.nbp,:]
            y = s.y[self.nbp:,:]
            x=np.transpose(x)
            y=np.transpose(y)

        elif self.choixInt==2:
            n = int(self.dt)
            N = int(self.T)
            self.dt = 2*np.pi/n
            self.T = 2*np.pi*N
            s = self.RungeKutta(Modelx,X0,self.T,self.dt)

            x = s[:self.nbp,0:N*n:n]
            y = s[self.nbp:,0:N*n:n]
            x=np.transpose(x)
            y=np.transpose(y)
            print(x.shape)

        return x,y