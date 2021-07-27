from tkinter import *
from tkinter.messagebox import *
from tkinter.filedialog import *
from Hamilton import Hamilton
from PotentielSpatial import PotentielSpatial
import matplotlib.pyplot as plt
#import random as rd
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
    if choixPhi.get() == 0:
        eta = float(entreta1.get())*(10**(int(entreta2.get())))
        h = Hamilton(eta,float(entrA.get()),float(entrrho.get()),int(entrnb.get()),int(entrM.get()),int(entrdx.get()),int(entrdy.get()),float(entrT.get()),float(entrdt.get()),choixInt.get(),choixOrdreGyro.get(),choixOrdreFLR.get(),int(entrnth.get()))
        x,y,nTrapped,slope = h.Dynamics()
        #plt.plot(x,y,'.')
        plt.plot(np.mod(x,2*np.pi),np.mod(y,2*np.pi),'.')
        #plt.title("Ordre Gyro =" + str(choixOrdreGyro.get()) + " , A = " + entrA.get() + " , rho = " + entrrho.get() + " , M = " + entrM.get() + " , eta = " + str(entreta1.get()) + "*10^" + str(entreta2.get()))
        if choixOrdreGyro.get()==2:
            titre = "Ordre Gyro =" + str(choixOrdreGyro.get()) + " , A = " + entrA.get() + " , rho = " + entrrho.get() + " , M = " + entrM.get() + " , eta = " + str(entreta1.get()) + "*10^" + str(entreta2.get())
            print("Gyro_" + str(choixOrdreGyro.get()) + "_A_" + entrA.get() + "_rho_" + entrrho.get() + "_M_" + entrM.get() + "_eta_" + str(entreta1.get()) + "_10^" + str(entreta2.get()))
        else:
            titre = "Ordre Gyro =" + str(choixOrdreGyro.get()) + " , A = " + entrA.get() + " , rho = " + entrrho.get() + " , M = " + entrM.get()
            print("Gyro_" + str(choixOrdreGyro.get()) + "_A_" + entrA.get() + "_rho_" + entrrho.get() + "_M_" + entrM.get())
        plt.title(titre)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig("change.png")
        #plt.savefig("Ordre Gyro =" + str(choixOrdreGyro.get()) + " , A = " + entrA.get() + " , rho = " + entrrho.get() + " , M = " + entrM.get() + " , eta = " + str(entreta1.get()) + "*10^" + str(entreta2.get()) + '.png')
        plt.show()
    else:
        eta = float(entreta1.get())*(10**(int(entreta2.get())))
        h = PotentielSpatial(eta,float(entrA.get()),float(entrrho.get()),int(entrnb.get()),int(entrM.get()),float(entrdx.get()),float(entrdy.get()),float(entrT.get()),float(entrdt.get()),choixInt.get(),choixOrdreGyro.get(),choixOrdreFLR.get(),int(entrnth.get()))
        x,y = h.Dynamics()
        #plt.plot(x,y,'.')
        plt.plot(np.mod(x,2*np.pi),np.mod(y,2*np.pi),'.')
        plt.title("Ordre Gyro =" + str(choixOrdreGyro.get()) + " , A = " + entrA.get() + " , rho = " + entrrho.get() + " , M = " + entrM.get() + " , eta = " + str(entreta1.get()) + "*10^" + str(entreta2.get()))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

fen1 = Tk()

#Label(fen1, text = 'B :').grid(padx = 5, pady = 2, sticky =E)
Label(fen1, text = 'eta :').grid(padx = 5, pady = 2, sticky =E)
Label(fen1, text = "nb particule test :").grid(padx = 2, pady = 5, sticky =E)
Label(fen1, text = 'A :').grid(padx = 2, pady = 5, sticky =E)
Label(fen1, text = 'rho :').grid(padx = 2, pady = 5, sticky =E)
Label(fen1, text = 'nombre de mode :').grid(padx = 2, pady = 5, sticky =E)
Label(fen1, text = 'Temps final :').grid(padx = 2, pady = 5, sticky =E)
Label(fen1, text = 'pas de temps :').grid(padx = 2, pady = 5, sticky =E)
Label(fen1, text = 'Taille grille x :').grid(padx = 2, pady = 5, sticky =E)
Label(fen1, text = 'Taille grille y :').grid(padx = 2, pady = 5, sticky =E)
#Label(fen1, text = 'nombre theta :').grid(padx = 2, pady = 5, sticky =E)
#entrB = Entry(fen1)
#entrB.insert(0,'1')
entreta1 = Entry(fen1,width = 5)
entreta1.insert(0,'1')
entreta2 = Entry(fen1,width = 5)
entreta2.insert(0,'-1')
entrnb = Entry(fen1)
entrnb.insert(0,'500')
entrA = Entry(fen1)
entrA.insert(0,'0.1')
entrrho = Entry(fen1)
entrrho.insert(0,'0')
entrM = Entry(fen1)
entrM.insert(0,'10')
entrT = Entry(fen1)
entrT.insert(0,'500')
entrdt = Entry(fen1)
entrdt.insert(0,'200')
entrdx = Entry(fen1)
entrdx.insert(0,'256')
entrdy = Entry(fen1)
entrdy.insert(0,'256')
entrnth = Entry(fen1)
entrnth.insert(0,'20')
#entrB.grid(row = 0, column = 1)
entreta1.grid(row = 0, column = 1,sticky = W)
Label(fen1, text = '*10^').grid(row=0,column = 1,padx = 5)
entreta2.grid(row = 0, column = 1,sticky = E)
entrnb.grid(row = 1, column = 1)
entrA.grid(row = 2, column = 1)
entrrho.grid(row = 3, column = 1)
entrM.grid(row = 4, column = 1)
entrT.grid(row = 5, column = 1)
entrdt.grid(row = 6, column = 1)
entrdx.grid(row = 7, column = 1)
entrdy.grid(row = 8, column = 1)
#entrnth.grid(row = 10, column = 1)

choixInt = IntVar()
choixInt.set(2)
Label(fen1, text = 'Methode integration').grid(row = 0, column = 3, padx = 2, pady = 2, columnspan = 2)
odeint = Radiobutton(fen1, text = 'odeint', variable = choixInt, value=0)
solve_ivp = Radiobutton(fen1, text = 'solve_ivp', variable = choixInt, value = 1)
runge = Radiobutton(fen1, text = 'runge-kutta ordre 4', variable = choixInt, value=2)
odeint.grid(row = 1, column = 3, padx = 2, pady = 5)
solve_ivp.grid(row = 2, column = 3, padx = 5, pady = 5)
runge.grid(row = 3, column = 3, padx = 2, pady = 5)

choixOrdreGyro = IntVar()
choixOrdreGyro.set(1)
Label(fen1, text = 'Ordre Gyro').grid(row = 0, column = 2, padx = 2, pady = 2, columnspan = 1)
ordre1Gyro = Radiobutton(fen1, text = 'Ordre 1', variable = choixOrdreGyro, value = 1)
ordre2Gyro = Radiobutton(fen1, text = 'Ordre 2', variable = choixOrdreGyro, value = 2)
ordre1Gyro.grid(row = 1, column = 2, padx = 2, pady = 5)
ordre2Gyro.grid(row = 2, column = 2, padx = 5, pady = 5)

choixOrdreFLR = IntVar()
choixOrdreFLR.set(1)
Label(fen1, text = 'Ordre FLR').grid(row = 5, column = 2, padx = 2, pady = 2, columnspan = 1)
ordre1FLR = Radiobutton(fen1, text = 'Tout ordre', variable = choixOrdreFLR, value = 1)
ordre2FLR = Radiobutton(fen1, text = 'Ordre 2', variable = choixOrdreFLR, value = 2)
ordre1FLR.grid(row = 6, column = 2, padx = 2, pady = 5)
ordre2FLR.grid(row = 7, column = 2, padx = 5, pady = 5)

choixPhi = IntVar()
choixPhi.set(0)
Label(fen1, text = 'Potentiel electrique').grid(row = 5, column = 3, padx = 2, pady = 2, columnspan = 2)
temporel = Radiobutton(fen1, text = 'temporel', variable = choixPhi, value=0)
spatial = Radiobutton(fen1, text = 'spatial', variable = choixPhi, value = 1)
temporel.grid(row = 6, column = 3, padx = 2, pady = 5)
spatial.grid(row = 7, column = 3, padx = 5, pady = 5)

bouton = Button(fen1, text ='Lancer simulation', command=recupHamilton)
bouton.grid(row = 10, columnspan = 4)

fen1.mainloop()