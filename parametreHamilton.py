from tkinter import *
from tkinter.messagebox import *
from tkinter.filedialog import *
from Hamilton import Hamilton
import matplotlib.pyplot as plt
import random as rd
from tqdm import tqdm
import numpy as np

def recupHamilton():
	"""
	for i in tqdm(range(int(entrnb.get()))):
		x0 = rd.uniform(-4,4)
		y0 = rd.uniform(0,8)
		#x0 = 0.2
		#y0 = 0.2
		h = Hamilton(float(entrB.get()),float(entrA.get()),float(entrrho.get()),int(entrnb.get()),int(entrM.get()),x0,y0,float(entrdx.get()),float(entrdy.get()),float(entrT.get()),float(entrdt.get()))
		#x,y,b = h.dynamics()
		x,y = h.dynamicsb()
		#plt.plot(x,y,'.')
		plt.plot(np.mod(x,2*np.pi),np.mod(y,2*np.pi),'.')
		#plt.plot(b)
	"""
	x0 = 0
	y0 = 0
	h = Hamilton(float(entrB.get()),float(entrA.get()),float(entrrho.get()),int(entrnb.get()),int(entrM.get()),x0,y0,float(entrdx.get()),float(entrdy.get()),float(entrT.get()),float(entrdt.get()),choixInt.get())
	x,y = h.dynamicsb()
	#plt.plot(x,y,'.')
	plt.plot(np.mod(x,2*np.pi),np.mod(y,2*np.pi),'.')
	#plt.xlim(-4,4)
	#plt.ylim(0,8)
	plt.title("A = " + entrA.get() + " , rho = " + entrrho.get() )
	plt.xlabel("x")
	plt.ylabel("y")
	plt.show()

fen1 = Tk()

Label(fen1, text = 'B').grid(padx = 5, pady = 2, sticky =E)
"""
Label(fen1, text = 'x0 :').grid(padx = 2, pady = 2, sticky =E)
Label(fen1, text = 'y0 :').grid(padx = 2, pady = 5, sticky =E)
"""
Label(fen1, text = "nb particule test :").grid(padx = 2, pady = 5, sticky =E)
Label(fen1, text = 'A :').grid(padx = 2, pady = 5, sticky =E)
Label(fen1, text = 'rho :').grid(padx = 2, pady = 5, sticky =E)
Label(fen1, text = 'nombre de mode :').grid(padx = 2, pady = 5, sticky =E)
Label(fen1, text = 'Temps final :').grid(padx = 2, pady = 5, sticky =E)
Label(fen1, text = 'pas de temps :').grid(padx = 2, pady = 5, sticky =E)
Label(fen1, text = 'dx :').grid(padx = 2, pady = 5, sticky =E)
Label(fen1, text = 'dy :').grid(padx = 2, pady = 5, sticky =E)
entrB = Entry(fen1)
entrB.insert(0,'1')
"""
entrx0 = Entry(fen1)
entrx0.insert(0,'0.1')
entry0 = Entry(fen1)
entry0.insert(0,'3')
"""
entrnb = Entry(fen1)
entrnb.insert(0,'1')
entrA = Entry(fen1)
entrA.insert(0,'0.4')
entrrho = Entry(fen1)
entrrho.insert(0,'0')
entrM = Entry(fen1)
entrM.insert(0,'1')
entrT = Entry(fen1)
entrT.insert(0,'10')
entrdt = Entry(fen1)
entrdt.insert(0,'0.0001')
entrdx = Entry(fen1)
entrdx.insert(0,'0.00001')
entrdy = Entry(fen1)
entrdy.insert(0,'0.00001')
entrB.grid(row = 0, column = 1)
"""
entrx0.grid(row = 1, column = 1)
entry0.grid(row = 2, column = 1)
"""
entrnb.grid(row = 1, column = 1)
entrA.grid(row = 2, column = 1)
entrrho.grid(row = 3, column = 1)
entrM.grid(row = 4, column = 1)
entrT.grid(row = 5, column = 1)
entrdt.grid(row = 6, column = 1)
entrdx.grid(row = 7, column = 1)
entrdy.grid(row = 8, column = 1)

choixInt = IntVar()
choixInt.set(1)
Label(fen1, text = 'Methode integration').grid(row = 0, column = 2, padx = 2, pady = 2, columnspan = 2)
odeint = Radiobutton(fen1, text = 'odeint', variable = choixInt, value=0)
solve_ivp = Radiobutton(fen1, text = 'solve_ivp', variable = choixInt, value = 1)
runge = Radiobutton(fen1, text = 'runge-kutta ordre 4', variable = choixInt, value=2)
leap = Radiobutton(fen1, text = 'leap-frog', variable = choixInt, value = 3)
odeint.grid(row = 1, column = 2, padx = 2, pady = 5)
solve_ivp.grid(row = 2, column = 2, padx = 5, pady = 5)
runge.grid(row = 3, column = 2, padx = 2, pady = 5)
leap.grid(row = 4, column = 2, padx = 5, pady = 5)

bouton = Button(fen1, text ='Lancer simulation', command=recupHamilton)
bouton.grid(row = 9, columnspan = 2)

fen1.mainloop()