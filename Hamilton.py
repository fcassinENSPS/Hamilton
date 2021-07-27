import numpy as np
#import scipy as sp
from scipy.integrate import odeint,solve_ivp
import scipy.special as spc
from scipy.stats import linregress
from tqdm import tqdm
import matplotlib.pyplot as plt

class Hamilton:
    def __init__(self,eta,A,rho,nbp,M,dx,dy,T,dt,choixInt,choixOrdreGyro,choixOrdreFLR,ntheta):
        # PARSING INPUTS 
        #self.B = B 			# champ magnétique B
        self.A = A
        self.Ab = A*spc.jv(0,rho*np.sqrt(2))		# amplitude Phi modifier
        self.rho = rho 		# rayon Larmor
        self.nbp = nbp		# nombres particules
        #self.M = M 			# nombre de mode champ electrostatique
        #self.x = 0 		# liste position particule dans le temps en x
        self.x = np.zeros(nbp)
        self.x[:] = np.random.rand(nbp)*2*np.pi
        #self.y = 0		# liste position particule dans le temps en y
        self.y = np.zeros(nbp)
        self.y[:] = np.random.rand(nbp)*2*np.pi
        self.T = T 				# Temps final
        self.dt = dt			# Nombre iteration dans le temps
        self.dx = dx		# taille maille x
        self.dy = dy 		# taille maille y 
        self.choixInt = choixInt
        self.choixOrdreGyro = choixOrdreGyro
        self.eta = eta

        if choixOrdreFLR == 2:
        	self.Ab = A*(1-rho**2/2)

        if rho == 0:
        	self.choixOrdreGyro = 0
        if (M%2)==0:
            self.M = M/2
        else:
            self.M =(M-1)/2

    def RungeKutta(self,Model,u0,Tf,dt):
        u = np.zeros((2*self.nbp,int(Tf/dt)+1))
        u[:,0] = u0
        for i in tqdm(range(int(Tf/dt))):
            du0 = Model(dt*i,u[:,i])
            du1 = Model(dt*(i+1/2),u[:,i]+dt*du0/2)
            du2 = Model(dt*(i+1/2),u[:,i]+dt*du1/2)
            du3 = Model(dt*(i+1),u[:,i]+dt*du2)
            u[:,i+1] = u[:,i] + dt*(du0+2*du1+2*du2+du3)/6
        return u

    def Potential0(self,x,y,t):
        cosx = np.cos(x)
        sinx = np.sin(x)
        cosy = np.cos(y)
        siny = np.sin(y)

        alpha = 1+2*np.cos((self.M+1)*t)*spc.eval_chebyu(self.M-1,np.cos(t))
        beta =2*np.cos(self.M*t)*spc.eval_chebyu(self.M-1,np.cos(t))

        dphidy = 2*np.pi*self.A*(beta*sinx*cosy-alpha*cosx*siny)
        dphidx = 2*np.pi*self.A*(beta*cosx*siny-alpha*sinx*cosy)

        return dphidy,dphidx

    def Potential1(self,x,y,t):
        cosx = np.cos(x)
        sinx = np.sin(x)
        cosy = np.cos(y)
        siny = np.sin(y)

        alpha = 1+2*np.cos((self.M+1)*t)*spc.eval_chebyu(self.M-1,np.cos(t))
        beta =2*np.cos(self.M*t)*spc.eval_chebyu(self.M-1,np.cos(t))

        dphidy = 2*np.pi*self.Ab*(beta*sinx*cosy-alpha*cosx*siny)
        dphidx = 2*np.pi*self.Ab*(beta*cosx*siny-alpha*sinx*cosy)

        return dphidy,dphidx

    def Potential2(self,x,y,t):
        cosx = np.cos(x)
        sinx = np.sin(x)
        cosy = np.cos(y)
        siny = np.sin(y)

        alpha = 1+2*np.cos((self.M+1)*t)*spc.eval_chebyu(self.M-1,np.cos(t))
        beta =2*np.cos(self.M*t)*spc.eval_chebyu(self.M-1,np.cos(t))

        dphidy = -2*np.pi*self.Ab*(beta*sinx*cosy-alpha*cosx*siny) + 2*self.A**2*self.eta*((alpha**2+beta**2)*np.cos(2*x)*siny*cosy-2*alpha*beta*sinx*cosx*np.cos(2*y))
        dphidx = 2*np.pi*self.Ab*(beta*cosx*siny-alpha*sinx*cosy) - 2*self.A**2*self.eta*((alpha**2+beta**2)*np.cos(2*y)*sinx*cosx-2*alpha*beta*siny*cosy*np.cos(2*x))

        return dphidy,dphidx

    def Model0(self,t,X):
        x = X[:self.nbp]
        y = X[self.nbp:]
        dXdt = np.zeros(2*self.nbp)

        dXdtb = self.Potential0(x,y,t)
        dXdt[:self.nbp] = -dXdtb[0]
        dXdt[self.nbp:] = dXdtb[1]

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

        dXdtb = self.Potential2(x,y,t)
        dXdt[:self.nbp] = dXdtb[0]
        dXdt[self.nbp:] = dXdtb[1]

        return dXdt

    def Dynamics(self):

        x = self.x
        y = self.y

        X0 = np.zeros(2*self.nbp)
        X0[:self.nbp] = x[:]
        X0[self.nbp:] = y[:]

        n = int(self.dt)
        N = int(self.T)

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
            self.dt = 2*np.pi/n
            self.T = 2*np.pi*N
            s = self.RungeKutta(Modelx,X0,self.T,self.dt)

            x = s[:self.nbp,0:N*n:n]
            y = s[self.nbp:,0:N*n:n]
            x=np.transpose(x)
            y=np.transpose(y)

        nTrapped = 0
        for i in range(self.nbp):
            if np.all((np.abs(x[0,i]-x[:,i])<2*np.pi)):
                nTrapped+=1
        #print('nTrapped')
        #print(nTrapped)
        """
        r2 = np.zeros(N)
        for i in range(N):
            r2[i] = 1/(self.T-i*self.dt)*np.mean(np.sum((x[0+i:N]-x[0:N-i])**2+(y[0+i:N]-y[0:N-i])**2,axis=1))
        t = np.arange(N)*2*np.pi
        slope, intercept, r, p, se = linregress(t, r2)
        """
        slope = 0
        #print('slope')
        #print(slope)
        #plt.plot(t,r2)
        #plt.savefig('r2.png')

        return x,y,nTrapped,slope