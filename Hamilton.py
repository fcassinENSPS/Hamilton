import numpy as np
import numpy.fft as fft
import scipy as sp
from scipy.integrate import odeint,solve_ivp,RK45
import scipy.special as spc
import matplotlib.pyplot as plt
from tqdm import tqdm

class Hamilton:
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
    def __init__(self,B,A,rho,nbp,M,dx,dy,T,dt,choixInt):
        # PARSING INPUTS 
        self.B = B 			# champ magnétique B
        self.A = A 			# amplitude Phi
        self.rho = rho 		# rayon Larmor
        self.nbp = nbp		# nombres particules
        self.M = M 			# nombre de mode champ electrostatique
        self.x = 0 		# liste position particule dans le temps en x
        self.xb = np.zeros((nbp,500))
        self.xb[:] = np.random.rand(nbp,1)*2*np.pi
        self.y = 0		# liste position particule dans le temps en y
        self.yb = np.zeros((nbp,500))
        self.yb[:] = np.random.rand(nbp,1)*2*np.pi
        self.T = T 				# Temps final
        self.dt = dt			# Nombre iteration dans le temps
        self.dx = dx		# taille maille x
        self.dy = dy 		# taille maille y 
        self.choixInt = choixInt
        #self.xnp = x0+y0
        #self.xnm = x0-y0
        self.wc = 15*self.B

        if (self.M%2)==0:
            self.Mt2n = np.linspace(-self.M,self.M,(self.M+1))
            self.Mt2n1 = np.linspace(-self.M+1,self.M-1,(self.M))

            #self.a2n = self.M+1
            #self.a2n1 = self.M
        else:
            self.Mt2n = np.linspace(-self.M+1,self.M-1,(self.M))
            self.Mt2n1 = np.linspace(-self.M,self.M,(self.M+1))

            #self.a2n = self.M
            #self.a2n1 = self.M+1
        self.a2n = 2*np.pi
        self.a2n1 = 2*np.pi


    def parameters(self,B,x0,yo,A):
    	self.B = B
    	self.A = A
    	self.x = [x0]
    	self.y = [yo]
    	self.xnp = [x0+y0]
    	self.xnm = [x0-y0]

    def rungekuttasimpson(self,model,u0,Tf,dt):
        u = np.zeros((2*self.nbp,int(Tf/dt)+1))
        u[:,0] = u0
        for i in tqdm(range(int(Tf/dt))):
            
            du0 = model(dt*i,u[:,i])
            du1 = model(dt*(i+1/2),u[:,i]+dt*du0/2)
            du2 = model(dt*(i+1/2),u[:,i]+dt*du1/2)
            du3 = model(dt*(i+1),u[:,i]+dt*du2)
            u[:,i+1] = u[:,i] + dt*(du0+2*du1+2*du2+du3)/6
            """
            du0 = dt*model(dt*i*2*np.pi,u[:,i])
            du1 = dt*model(dt*(i+1/4)*2*np.pi,u[:,i]+du0/4)
            du2 = dt*model(dt*(i+3/8)*2*np.pi,u[:,i]+du0*3/32+9/32*du1)
            du3 = dt*model(dt*(i+12/13)*2*np.pi,u[:,i]+1932/2197*du0-7200/2197*du1+7296/2197*du2)
            du4 = dt*model(dt*(i+1)*2*np.pi,u[:,i]+439/216*du0-8*du1+3680/513*du2-845/4104*du3)
            #du5 = dt*model(dt*(i+1/2),u[:,i]-8/27*du0+2*du1-3544/2565*du2+1859/4104*du3-11/40*du4)
            u[:,i+1] = u[:,i] + 25/216*du0 + 1408/2565*du2 +2197/4101*du3-1/5*du4
            #u[:,i+1] = u[:,i] + 16/135*du0 + 6656/12.825*du2 +28.561/56.430*du3-9/50*du4+2/55*du5
            """
        return u

    def leapfrog(self,model,u0,v0,Tf,dt):
        u = np.zeros((2*self.nbp,int(Tf/dt)+1))
        v = np.zeros((2*self.nbp,int(Tf/dt)+1))
        u[:,0]=u0
        v[:,0]=v0[:,0]
        for i in tqdm(range(int(Tf/dt))):
    	    a = model(dt*i,u[:,i])
    	    v[:self.nbp,i+1] = v[:self.nbp,i] + self.wc*a[self.nbp:]
    	    v[self.nbp:,i+1] = v[self.nbp:,i] - self.wc*a[:self.nbp]
    	    u[:,i+1] = u[:,i] + v[:,i]*dt
        return u
    
    """
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

        return phi,dphidy,dphidx
    """

    def potentialb(self,x,y,t):
        A = self.A
        #print(t)

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

            y2n,Mt2n = np.meshgrid(y,Mt2n)
            y2n1,Mt2n1 = np.meshgrid(y,Mt2n1)
            y1 = y2n-Mt2n
            y2 = y2n1-Mt2n1
            Ytheta1 = np.zeros((theta.shape[0],y1.shape[0],y1.shape[1]))
            Ytheta2 = np.zeros((theta.shape[0],y2.shape[0],y2.shape[1]))
            Ytheta1[:,:] = y1
            Ytheta2[:,:] = y2
            Y1b = Ytheta1.transpose(1,2,0)
            Y2b = Ytheta2.transpose(1,2,0)
            #print(Y1b.shape)
            Ytheta1 = np.sin((Y1b-rhos).transpose(2,0,1))
            Ytheta2 = np.sin((Y2b-rhos).transpose(2,0,1))
            #print(Ytheta1.shape)

            #print(np.mean(cosx*np.sum(np.sin((Y1b-rhos).transpose(2,0,1)),axis=1)).shape)

            da2ndy = -(self.a2n)*A*np.mean(cosx*np.sum(np.sin((Y1b-rhos).transpose(2,0,1)),axis=1),axis=0)
            da2ndx = -(self.a2n)*A*np.mean(sinx*np.sum(np.cos((Y1b-rhos).transpose(2,0,1)),axis=1),axis=0)

            da2n1dy = self.a2n1*A*np.mean(cosx*np.sum(np.sin((Y2b-rhos).transpose(2,0,1)),axis=1),axis=0)
            da2n1dx = self.a2n1*A*np.mean(sinx*np.sum(np.cos((Y2b-rhos).transpose(2,0,1)),axis=1),axis=0)

        return (da2ndy+da2n1dy),(da2ndx+da2n1dx)

    def modelx(self,t,X):
        x = X[:self.nbp]
        y = X[self.nbp:]
        dXdt = np.zeros(2*self.nbp)

        dXdtb = self.potentialb(x,y,t)
    	#dXdt = [-(self.potential(x,y+dy,t)-self.potential(x,y-dy,t)/(2*dy)),(self.potential(x+dx,y,t)-self.potential(x-dx,y,t)/(2*dx))]
    	#dXdt = [-self.potential(x,y,t)[1],self.potential(x,y,t)[2]]
        dXdt[:self.nbp] = -dXdtb[0]
        dXdt[self.nbp:] = dXdtb[1]

        return dXdt

    def dynamicsb(self):

        x = self.x
        y = self.y
        xb = self.xb
        yb = self.yb
        dx = self.dx
        dy = self.dy

        #X0 = [x[0],y[0]]
        X0b = np.zeros(2*self.nbp)
        X0b[:self.nbp] = xb[:,0]
        X0b[self.nbp:] = yb[:,0]
        if self.choixInt==0:
            t = 2*np.pi*np.arange(self.T)

            s = odeint(self.modelx,X0b,t,atol=10**(-6),rtol=10**(-6))

            x = s.y[:,:self.nbp]
            y = s.y[:,self.nbp:]

        elif self.choixInt==1:
            t = 2*np.pi*np.arange(int(self.T))

            s = solve_ivp(self.modelx,(0,2*np.pi*int(self.T)),X0b,t_eval=t,atol=10**(-6),rtol=10**(-6))

            x = s.y[:self.nbp,:]
            y = s.y[self.nbp:,:]
            x=np.transpose(x)
            y=np.transpose(y)

        elif self.choixInt==2:
            s = self.rungekuttasimpson(self.modelx,X0b,self.T,self.dt)

            x = s[:self.nbp,0:int(self.T/self.dt):int(2*np.pi)]
            y = s[self.nbp:,0:int(self.T/self.dt):int(2*np.pi)]
            x=np.transpose(x)
            y=np.transpose(y)

        elif self.choixInt==3:
            v0 = np.zeros(2*self.nbp)
            v0 = np.random.rand(2*self.nbp,1)*2*np.pi

            s = self.leapfrog(self.modelx,X0b,v0,self.T,self.dt)

            x = s[:self.nbp,0:int(self.T/self.dt):int(2*np.pi)]
            y = s[self.nbp:,0:int(self.T/self.dt):int(2*np.pi)]
            x=np.transpose(x)
            y=np.transpose(y)

        return x,y

    def dynamics(self):
        for i in range(int(self.T/self.dt)):
            if self.rho==0:
            	self.xnp = self.xnp - 2*np.pi*self.A*np.sin(self.xnm)	#(x+y)
            	self.xnm = self.xnm + 2*np.pi*self.A*np.sin(self.xnp)	#(x-y)
            else:
            	self.xnp = self.xnp - 2*np.pi*self.A*spc.jv(0,self.rho*np.sqrt(2))*np.sin(self.xnm)	#(x+y)
            	self.xnm = self.xnm + 2*np.pi*self.A*spc.jv(0,self.rho*np.sqrt(2))*np.sin(self.xnp)	#(x-y)

            self.x.append((self.xnp + self.xnm)/2)
            self.y.append((self.xnp - self.xnm)/2)

        return(self.x,self.y)

