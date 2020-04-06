import numpy as np
import math as m
import matplotlib.pyplot as plt

y_sweep_0, cl_sweep_0 = np.loadtxt("Spanload_A2.50_sweep_0.dat", skiprows=1, unpack=True, usecols=[0,1])
CL_sweep_0 = sum(cl_sweep_0)/len(y_sweep_0)
print(CL_sweep_0)

Valeur_x_sweep_0 = np.zeros(len(y_sweep_0))
Valeur_y_sweep_0 = np.zeros(len(y_sweep_0))
for i in range(len(y_sweep_0)):
    Valeur_x_sweep_0[i] = y_sweep_0[i]/y_sweep_0[-1]
    Valeur_y_sweep_0[i] = cl_sweep_0[i]/CL_sweep_0


# Visualisation graphique
plt.plot(Valeur_x_sweep_0, Valeur_y_sweep_0, color="green", label='Sweep = 0°')

#-------------------------------------------------------------------------------------------------------

y_sweep_45, cl_sweep_45 = np.loadtxt("Spanload_A2.50_sweep_45.dat", skiprows=1, unpack=True, usecols=[0,1])
CL_sweep_45 = sum(cl_sweep_45)/len(y_sweep_45)
print(CL_sweep_45)

Valeur_x_sweep_45 = np.zeros(len(y_sweep_45))
Valeur_y_sweep_45 = np.zeros(len(y_sweep_45))
for i in range(len(y_sweep_45)):
    Valeur_x_sweep_45[i] = y_sweep_45[i]/y_sweep_45[-1]
    Valeur_y_sweep_45[i] = cl_sweep_45[i]/CL_sweep_45


# Visualisation graphique
plt.plot(Valeur_x_sweep_45, Valeur_y_sweep_45, color="blue", label='Sweep = 45°')

#-------------------------------------------------------------------------------------------------------

y_sweep_135, cl_sweep_135 = np.loadtxt("Spanload_A2.50_sweep_135.dat", skiprows=1, unpack=True, usecols=[0,1])
CL_sweep_135 = sum(cl_sweep_135)/len(y_sweep_135)
print(CL_sweep_135)

Valeur_x_sweep_135 = np.zeros(len(y_sweep_135))
Valeur_y_sweep_135 = np.zeros(len(y_sweep_135))
for i in range(len(y_sweep_135)):
    Valeur_x_sweep_135[i] = y_sweep_135[i]/y_sweep_135[-1]
    Valeur_y_sweep_135[i] = cl_sweep_135[i]/CL_sweep_135

    
# Visualisation graphique
plt.plot(Valeur_x_sweep_135, Valeur_y_sweep_135, color="black", label='Sweep = 135°')
# Propriétés des graphiques
plt.xlim(0, 1.0)
plt.ylim(0, 1.5)
#plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('y/b')
plt.ylabel('Cl/CL')
plt.title('Distribution de la portance')
plt.legend(framealpha=1, frameon=True);




plt.show()
