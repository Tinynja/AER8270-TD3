import numpy as np
import math as m
import matplotlib.pyplot as plt

y_lambda_1, cl_lambda_1 = np.loadtxt("Spanload_A2.50_lambda_1.dat", skiprows=1, unpack=True, usecols=[0,1])
CL_lambda_1 = sum(cl_lambda_1)/len(y_lambda_1)
print(CL_lambda_1)

Valeur_x_lambda_1 = np.zeros(len(y_lambda_1))
Valeur_y_lambda_1 = np.zeros(len(y_lambda_1))
for i in range(len(y_lambda_1)):
    Valeur_x_lambda_1[i] = y_lambda_1[i]/y_lambda_1[-1]
    Valeur_y_lambda_1[i] = cl_lambda_1[i]/CL_lambda_1


# Visualisation graphique
plt.plot(Valeur_x_lambda_1, Valeur_y_lambda_1, color="green", label='lambda = 1.0')

#-------------------------------------------------------------------------------------------------------

y_lambda_60, cl_lambda_60 = np.loadtxt("Spanload_A2.50_lambda_0.60.dat", skiprows=1, unpack=True, usecols=[0,1])
CL_lambda_60 = sum(cl_lambda_60)/len(y_lambda_60)
print(CL_lambda_60)

Valeur_x_lambda_60 = np.zeros(len(y_lambda_60))
Valeur_y_lambda_60 = np.zeros(len(y_lambda_60))
for i in range(len(y_lambda_60)):
    Valeur_x_lambda_60[i] = y_lambda_60[i]/y_lambda_60[-1]
    Valeur_y_lambda_60[i] = cl_lambda_60[i]/CL_lambda_60


# Visualisation graphique
plt.plot(Valeur_x_lambda_60, Valeur_y_lambda_60, color="blue", label='lambda = 0.60')

#-------------------------------------------------------------------------------------------------------

y_lambda_40, cl_lambda_40 = np.loadtxt("Spanload_A2.50_lambda_0.40.dat", skiprows=1, unpack=True, usecols=[0,1])
CL_lambda_40 = sum(cl_lambda_40)/len(y_lambda_40)
print(CL_lambda_40)

Valeur_x_lambda_40 = np.zeros(len(y_lambda_40))
Valeur_y_lambda_40 = np.zeros(len(y_lambda_40))
for i in range(len(y_lambda_40)):
    Valeur_x_lambda_40[i] = y_lambda_40[i]/y_lambda_40[-1]
    Valeur_y_lambda_40[i] = cl_lambda_40[i]/CL_lambda_40


# Visualisation graphique
plt.plot(Valeur_x_lambda_40, Valeur_y_lambda_40, color="black", label='lambda = 0.40')


#-------------------------------------------------------------------------------------------------------

y_lambda_0, cl_lambda_0 = np.loadtxt("Spanload_A2.50_lambda_0.0.dat", skiprows=1, unpack=True, usecols=[0,1])
CL_lambda_0 = sum(cl_lambda_0)/len(y_lambda_0)
print(CL_lambda_0)

Valeur_x_lambda_0 = np.zeros(len(y_lambda_0))
Valeur_y_lambda_0 = np.zeros(len(y_lambda_0))
for i in range(len(y_lambda_0)):
    Valeur_x_lambda_0[i] = y_lambda_0[i]/y_lambda_0[-1]
    Valeur_y_lambda_0[i] = cl_lambda_0[i]/CL_lambda_0


# Visualisation graphique
plt.plot(Valeur_x_lambda_0, Valeur_y_lambda_0, color="orange", label='lambda = 0.0')

# Propriétés des graphiques
plt.xlim(0, 1.0)
plt.ylim(0, 1.5)
#plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('y/b')
plt.ylabel('Cl/CL')
plt.title('Distribution de la portance')
plt.legend(framealpha=1, frameon=True);




plt.show()
