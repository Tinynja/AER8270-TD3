import numpy as np
import math as m
import matplotlib.pyplot as plt

y_AR_1, cl_AR_1 = np.loadtxt("Spanload_A2.50_AR_1.0.dat", skiprows=1, unpack=True, usecols=[0,1])
CL_AR_1 = sum(cl_AR_1)/len(y_AR_1)
print(CL_AR_1)

Valeur_x_AR_1 = np.zeros(len(y_AR_1))
Valeur_y_AR_1 = np.zeros(len(y_AR_1))
for i in range(len(y_AR_1)):
    Valeur_x_AR_1[i] = y_AR_1[i]/y_AR_1[-1]
    Valeur_y_AR_1[i] = cl_AR_1[i]/CL_AR_1
    
# Coefficient d'Oswald
CD_AR_1 = CL_AR_1**2/(np.pi*1.0*1.0)
print('Pour AR = 1, CD = %3lf' %CD_AR_1)

# Visualisation graphique
plt.plot(Valeur_x_AR_1, Valeur_y_AR_1, color="green", label='AR = 1.0')

#-------------------------------------------------------------------------------------------------------

y_AR_2, cl_AR_2 = np.loadtxt("Spanload_A2.50_AR_2.0.dat", skiprows=1, unpack=True, usecols=[0,1])
CL_AR_2 = sum(cl_AR_2)/len(y_AR_2)
print(CL_AR_2)

Valeur_x_AR_2 = np.zeros(len(y_AR_2))
Valeur_y_AR_2 = np.zeros(len(y_AR_2))
for i in range(len(y_AR_2)):
    Valeur_x_AR_2[i] = y_AR_2[i]/y_AR_2[-1]
    Valeur_y_AR_2[i] = cl_AR_2[i]/CL_AR_2

# Coefficient d'Oswald
CD_AR_2 = CL_AR_2**2/(np.pi*1.0*2.0)
print('Pour AR = 2, CD = %3lf' %CD_AR_2)

# Visualisation graphique
plt.plot(Valeur_x_AR_2, Valeur_y_AR_2, color="blue", label='AR = 2.0')

#-------------------------------------------------------------------------------------------------------

y_AR_3, cl_AR_3 = np.loadtxt("Spanload_A2.50_AR_3.0.dat", skiprows=1, unpack=True, usecols=[0,1])
CL_AR_3 = sum(cl_AR_3)/len(y_AR_3)
print(CL_AR_3)

Valeur_x_AR_3 = np.zeros(len(y_AR_3))
Valeur_y_AR_3 = np.zeros(len(y_AR_3))
for i in range(len(y_AR_3)):
    Valeur_x_AR_3[i] = y_AR_3[i]/y_AR_3[-1]
    Valeur_y_AR_3[i] = cl_AR_3[i]/CL_AR_3

# Coefficient d'Oswald
CD_AR_3 = CL_AR_3**2/(np.pi*1.0*3.0)
print('Pour AR = 3, CD = %3lf' %CD_AR_3)

# Visualisation graphique
plt.plot(Valeur_x_AR_3, Valeur_y_AR_3, color="black", label='AR = 3.0')

#-------------------------------------------------------------------------------------------------------

y_AR_4, cl_AR_4 = np.loadtxt("Spanload_A2.50_AR_4.0.dat", skiprows=1, unpack=True, usecols=[0,1])
CL_AR_4 = sum(cl_AR_4)/len(y_AR_4)
print(CL_AR_4)

Valeur_x_AR_4 = np.zeros(len(y_AR_4))
Valeur_y_AR_4 = np.zeros(len(y_AR_4))
for i in range(len(y_AR_4)):
    Valeur_x_AR_4[i] = y_AR_4[i]/y_AR_4[-1]
    Valeur_y_AR_4[i] = cl_AR_4[i]/CL_AR_4

# Coefficient d'Oswald
CD_AR_4 = CL_AR_4**2/(np.pi*1.0*4.0)
print('Pour AR = 4, CD = %3lf' %CD_AR_4)

# Visualisation graphique
plt.plot(Valeur_x_AR_4, Valeur_y_AR_4, color="red", label='AR = 4.0')

#-------------------------------------------------------------------------------------------------------

y_AR_5, cl_AR_5 = np.loadtxt("Spanload_A2.50_AR_5.0.dat", skiprows=1, unpack=True, usecols=[0,1])
CL_AR_5 = sum(cl_AR_5)/len(y_AR_5)
print(CL_AR_5)

Valeur_x_AR_5 = np.zeros(len(y_AR_5))
Valeur_y_AR_5 = np.zeros(len(y_AR_5))
for i in range(len(y_AR_5)):
    Valeur_x_AR_5[i] = y_AR_5[i]/y_AR_5[-1]
    Valeur_y_AR_5[i] = cl_AR_5[i]/CL_AR_5

# Coefficient d'Oswald
CD_AR_5 = CL_AR_5**2/(np.pi*1.0*5.0)
print('Pour AR = 5, CD = %3lf' %CD_AR_5)

# Visualisation graphique
plt.plot(Valeur_x_AR_5, Valeur_y_AR_5, color="yellow", label='AR = 5.0')

#-------------------------------------------------------------------------------------------------------

y_AR_6, cl_AR_6 = np.loadtxt("Spanload_A2.50_AR_6.0.dat", skiprows=1, unpack=True, usecols=[0,1])
CL_AR_6 = sum(cl_AR_6)/len(y_AR_6)
print(CL_AR_6)

Valeur_x_AR_6 = np.zeros(len(y_AR_6))
Valeur_y_AR_6 = np.zeros(len(y_AR_6))
for i in range(len(y_AR_6)):
    Valeur_x_AR_6[i] = y_AR_6[i]/y_AR_6[-1]
    Valeur_y_AR_6[i] = cl_AR_6[i]/CL_AR_6

# Coefficient d'Oswald
CD_AR_6 = CL_AR_6**2/(np.pi*1.0*6.0)
print('Pour AR = 6, CD = %3lf' %CD_AR_6)

# Visualisation graphique
plt.plot(Valeur_x_AR_6, Valeur_y_AR_6, color="pink", label='AR = 6.0')

#-------------------------------------------------------------------------------------------------------

y_AR_7, cl_AR_7 = np.loadtxt("Spanload_A2.50_AR_7.0.dat", skiprows=1, unpack=True, usecols=[0,1])
CL_AR_7 = sum(cl_AR_7)/len(y_AR_7)
print(CL_AR_7)

Valeur_x_AR_7 = np.zeros(len(y_AR_7))
Valeur_y_AR_7 = np.zeros(len(y_AR_7))
for i in range(len(y_AR_7)):
    Valeur_x_AR_7[i] = y_AR_7[i]/y_AR_7[-1]
    Valeur_y_AR_7[i] = cl_AR_7[i]/CL_AR_7

# Coefficient d'Oswald
CD_AR_7 = CL_AR_7**2/(np.pi*1.0*7.0)
print('Pour AR = 7, CD = %3lf' %CD_AR_7)

# Visualisation graphique
plt.plot(Valeur_x_AR_7, Valeur_y_AR_7, color="orange", label='AR = 7.0')
# Propriétés des graphiques
plt.xlim(0, 1.0)
plt.ylim(0, 1.5)
#plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('y/b')
plt.ylabel('Cl/CL')
plt.title('Distribution de la portance selon différents AR')
plt.legend(framealpha=1, frameon=True);

#-------------------------------------------------------------------------------------------------------






plt.show()
