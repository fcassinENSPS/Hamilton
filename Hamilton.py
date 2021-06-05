import numpy as np
import numpy.fft as fft
import scipy as sp
from scipy.integrate import odeint,solve_ivp
import scipy.special as spc
import matplotlib.pyplot as plt
from tqdm import tqdm

class Hamilton:
    """
	Hamilton class gathering the potential calculation and dynamics
	Fourier class gathering the PSD calculation for PSF reconstruction. 
    Inputs are:
        - B magnetic field
        - e sign charge
        - eta 
        - w0
        - k
        - phi1
    """
    def __init__(self,B,A,rho,nbp,M,x0,y0,dx,dy,T,dt):
        # PARSING INPUTS 
        self.B = B 			# champ magnétique B
        self.A = A 			# amplitude Phi
        self.rho = rho 		# rayon Larmor
        self.nbp = nbp		# nombres particules
        self.M = M 			# nombre de mode champ electrostatique
        self.x = [x0,x0] 		# liste position particule dans le temps en x
        self.xb = np.zeros((nbp,500))
        self.xb[:] = np.random.rand(nbp,1)*2*np.pi
        #a = np.arange(500)
        #xb = np.linspace(0,nbp+1,num=nbp)/nbp*2*np.pi
        #xb,ax = np.meshgrid(xb,a)
        #self.xb = np.transpose(xb) 
        self.y = [y0,y0] 		# liste position particule dans le temps en y
        self.yb = np.zeros((nbp,500))
        self.yb[:] = np.random.rand(nbp,1)*2*np.pi
        #yb = np.linspace(1,nbp+1.5,num=nbp)/nbp*2*np.pi
        #yb,ax = np.meshgrid(yb,a)
        #self.yb = np.transpose(yb)
        self.T = T 				# Temps final
        self.dt = dt			# Nombre iteration dans le temps
        self.dx = dx		# taille maille x
        self.dy = dy 		# taille maille y 
        self.xnp = x0+y0
        self.xnm = x0-y0
        self.i = 0
        if (self.M%2)==0:
            self.Mt2n = np.linspace(-self.M,self.M,(2*self.M+1))
            self.Mt2n1 = np.linspace(-self.M+1,self.M-1,(self.M))

            self.a2n = self.M+1
            self.a2n1 = self.M
        else:
            self.Mt2n = np.linspace(-self.M+1,self.M-1,(self.M))
            self.Mt2n1 = np.linspace(-self.M,self.M,(self.M+1))

            self.a2n = self.M
            self.a2n1 = self.M+1


    def parameters(self,B,x0,yo,A):
    	self.B = B
    	self.A = A
    	self.x = [x0]
    	self.y = [yo]
    	self.xnp = [x0+y0]
    	self.xnm = [x0-y0]

    
    def potential(self,x,y,t):
		#A = self.k**2/self.w0*self.phi1/self.B
        A = self.A

        phi = A*np.cos(x)*np.cos(y)
        dphidy = -A*np.cos(x)*np.sin(y)
        dphidx = -A*np.sin(x)*np.cos(y)
        for i in range(self.M):
        	if (i+1)%2 ==0:
        		#phi += A*(np.cos(x)*np.cos(y-(i+1)*t)+np.cos(x)*np.cos(y+(i+1)*t))
        		dphidy += -A*(np.cos(x)*np.sin(y-(i+1)*t)+np.cos(x)*np.sin(y+(i+1)*t))
        		dphidx += -A*(np.sin(x)*np.cos(y-(i+1)*t)+np.sin(x)*np.cos(y+(i+1)*t))
        		#phi += A*(np.cos(x)*np.cos(y-(i+1)*t))
        	else:
        		#phi += A*(np.cos(x+np.pi/2)*np.cos(y+np.pi/2-(i+1)*t)+np.cos(x+np.pi/2)*np.cos(y+np.pi/2+(i+1)*t))
        		dphidy += -A*(np.cos(x+np.pi/2.0)*np.sin(y+np.pi/2.0-float((i+1))*t)+np.cos(x+np.pi/2.0)*np.sin(y+np.pi/2.0+float((i+1))*t))
        		dphidx += -A*(np.sin(x+np.pi/2.0)*np.cos(y+np.pi/2.0-float((i+1))*t)+np.sin(x+np.pi/2.0)*np.cos(y+np.pi/2.0+float((i+1))*t))
        		#phi += A*(np.cos(x+np.pi/2)*np.cos(y+np.pi/2-(i+1)*t))

        """
		kx = np.linspace(-x,x,Nx)	#
		ky = np.linspace(-y,y,Ny)	# grille spatial pour phi
		kx,ky = np.meshgrid(kx,ky)	#
		t = np.linspace(-100,100,201)*T
		
		somme = kx*0

        t = np.linspace(-400,400,801)*self.dt
        somme = 0
        
        for i in t:
        	somme += np.cos(x)*np.cos(y-2*i) + np.cos(x+np.pi/2)*np.cos(y+np.pi/2-(2*i+1))
        
        somme = (np.cos(x+y)+np.cos(x-y))
        
        phi = A *np.pi* somme
        """
        return phi,dphidy,dphidx

    def potentialb(self,x,y,t):
        A = self.A

        
        """
        Mt2n = np.linspace(-2*self.M,2*self.M,(2*self.M+1))*t
        Mt2n1 = np.linspace(-2*self.M+1,2*self.M-1,(2*self.M))*t
        
        if (self.M%2)==0:
            Mt2n = np.linspace(-self.M,self.M,(2*self.M+1))*t
            Mt2n1 = np.linspace(-self.M+1,self.M-1,(self.M))*t
        else:
            Mt2n = np.linspace(-self.M+1,self.M-1,(self.M))*t
            Mt2n1 = np.linspace(-self.M,self.M,(self.M+1))*t
        """
        Mt2n = self.Mt2n*t
        Mt2n1 = self.Mt2n1*t

        if self.rho==0:
            cosx = np.cos(x)
            sinx = np.sin(x)

            y2n,Mt2n = np.meshgrid(y,Mt2n)
            y2n1,Mt2n1 = np.meshgrid(y,Mt2n1)

            da2ndy = -(self.a2n)*A*cosx*(np.sum(np.sin(y2n-Mt2n),axis=0))
            da2ndx = -(self.a2n)*A*sinx*(np.sum(np.cos(y2n-Mt2n),axis=0))

            da2n1dy = self.a2n1*A*sinx*(np.sum(np.cos(y2n1-Mt2n1),axis=0))
            da2n1dx = self.a2n1*A*cosx*(np.sum(np.sin(y2n1-Mt2n1),axis=0))
        else:
            theta = np.linspace(0,2*np.pi,10)
            rhoc = self.rho*np.cos(theta)
            rhos = self.rho*np.sin(theta)
            xb,rhocb = np.meshgrid(x,rhoc)

            cosx = np.cos(xb + rhocb)
            sinx = np.sin(xb + rhocb)

            yb,rhosb = np.meshgrid(y,rhos)

        return (da2ndy+da2n1dy),(da2ndx+da2n1dx)

    def modelx(self,t,X):
        """
    	x,y = X
    	dy = self.dy
    	dx = self.dx
        """
        self.i+=1

        x = X[:self.nbp]
        y = X[self.nbp:]
        dXdt = np.zeros(2*self.nbp)


    	#dXdt = [-(self.potential(x,y+dy,t)-self.potential(x,y-dy,t)/(2*dy)),(self.potential(x+dx,y,t)-self.potential(x-dx,y,t)/(2*dx))]
    	#dXdt = [-self.potential(x,y,t)[1],self.potential(x,y,t)[2]]
        dXdt[:self.nbp] = -self.potentialb(x,y,t)[0]
        dXdt[self.nbp:] = self.potentialb(x,y,t)[1]

        print(self.i)

        return dXdt

    """
	def fluctualpotential(phi,av):
		return phi-av

	def average():

		theta = np.arc
		av = 1/(2*np.pi)
		return av
    """
    def dynamicsb(self):

    	#phi = potential(self.nbx,self.nby,self.Nx,self.Ny)
        #self.dt = 2*np.pi
        x = self.x
        y = self.y
        xb = self.xb
        yb = self.yb
        dx = self.dx
        dy = self.dy

        X0 = [x[0],y[0]]
        X0b = np.zeros(2*self.nbp)
        X0b[:self.nbp] = xb[:,0]
        X0b[self.nbp:] = yb[:,0]

        #t = np.linspace(0,self.T,2*np.pi*int(self.T/self.dt))
        t = 2*np.pi*np.arange(500)

        s = solve_ivp(self.modelx,(0,2*np.pi*500),X0b,t_eval=t,atol=10**(-6),rtol=10**(-6))

        x = s.y[:self.nbp,:]
        y = s.y[self.nbp:,:]

        """
        bahx = []
        bahxb =[]

        for i in range(1000):
        #self.x.append(self.x[i]-phi[x[i],y[i+1]]-phi[x[i],y[i]])
            #bahx.append((self.potential(x[i],y[i]+dy)-self.potential(x[i],y[i]-dy))/(2*dy))
            #bahxb.append(np.pi*self.A*(np.sin(self.xnm)-np.sin(self.xnp)))
            #x.append(x[i]-(self.potential(x[i],y[i]+dy)-self.potential(x[i],y[i]-dy))/(2*dy))
            #y.append(y[i]+(self.potential(x[i]+dx,y[i])-self.potential(x[i]-dx,y[i]))/(2*dx))
            if self.rho ==0:
                self.xnp = self.xnp + np.pi*self.A*((np.cos(self.xnm+dx)-np.cos(self.xnm-dx))/(2*dx)-(np.cos(self.xnm-dy)-np.cos(self.xnm+dy))/(2*dy))
                self.xnm = self.xnm - np.pi*self.A*((np.cos(self.xnp+dx)-np.cos(self.xnp-dx))/(2*dx)+(np.cos(self.xnp+dy)-np.cos(self.xnp-dy))/(2*dy))
            else:
                self.xnp = self.xnp + np.pi*self.A*spc.jv(0,self.rho*np.sqrt(2))*((np.cos(self.xnm+dx)-np.cos(self.xnm-dx))/(2*dx)-(np.cos(self.xnm-dy)-np.cos(self.xnm+dy))/(2*dy))
                self.xnm = self.xnm - np.pi*self.A*spc.jv(0,self.rho*np.sqrt(2))*((np.cos(self.xnp+dx)-np.cos(self.xnp-dx))/(2*dx)+(np.cos(self.xnp+dy)-np.cos(self.xnp-dy))/(2*dy))
            x.append((self.xnp+self.xnm)/2)
            y.append((self.xnp-self.xnm)/2)
        """



        return x,y

    def dynamics(self):
        """
		self.xnp = self.xnp - 2*np.pi*A*spc.jv(np.sqrt(2)*rho)*np.sin(xnm)	#(x+y+rho)
		self.xnm = self.xnm + 2*np.pi*A*spc.jv(np.sqrt(2)*rho)*np.sin(xnp)	#(x-y+rho)
         """
        bahx = []
        for i in range(int(self.T/self.dt)):
            if self.rho==0:
            	self.xnp = self.xnp - 2*np.pi*self.A*np.sin(self.xnm)	#(x+y)
            	self.xnm = self.xnm + 2*np.pi*self.A*np.sin(self.xnp)	#(x-y)
            else:
            	self.xnp = self.xnp - 2*np.pi*self.A*spc.jv(0,self.rho*np.sqrt(2))*np.sin(self.xnm)	#(x+y)
            	self.xnm = self.xnm + 2*np.pi*self.A*spc.jv(0,self.rho*np.sqrt(2))*np.sin(self.xnp)	#(x-y)

            bahx.append(np.pi*self.A*np.sin(self.xnm))

            self.x.append((self.xnp + self.xnm)/2)
            self.y.append((self.xnp - self.xnm)/2)

        return(self.x,self.y,bahx)

