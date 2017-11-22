#IMPORTS FOR PROJECT
# Import pandas
import pandas as pd
# Import matplotlib
import matplotlib.pyplot as plt
# Import numpy
import numpy as np

# Read in white wine data
#white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')
white = pd.read_csv("C:\\Users\\Daniel\\Documents\\Unal\\EjemploKeras\\negativo.csv", sep=';')


# Read in red wine data
#red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
red = pd.read_csv("C:\\Users\\Daniel\\Documents\\Unal\\EjemploKeras\\positivo.csv", sep=';')


#COMPROBACION  DE LOS DATOS CON FUNCIONES DE PANDAS:
# Print info on white wine
#print("vino Blanco: ")
#print(white.info())

# Print info on red wine
#print("vino Rojo: ")
#print(red.info())

# First rows of `red`
#print(red.head())

# Last rows of `white`
#print(white.tail())

# Take a sample of 5 rows of `red`
#print(red.sample(5))

# Describe `white`
#print(white.describe())

# Double check for null values in `red`
#print(pd.isnull(red))

fig, ax = plt.subplots(1, 2)

ax[0].hist(red.Altura_P_N_Mar, 10, facecolor='red', alpha=0.5, label="Red wine")
ax[1].hist(white.Altura_P_N_Mar, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="White wine")

fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
ax[0].set_ylim([0, 20])
ax[0].set_xlabel("PROMEDIO Altura Nivel del mar")
ax[0].set_ylabel("Frequency")
ax[1].set_xlabel("PROMEDIO Altura Nivel del mar")
ax[1].set_ylabel("Frequency")
#ax[0].legend(loc='best')
#ax[1].legend(loc='best')
fig.suptitle("Distribution of PROMEDIO Altura Nivel del mar mts")

plt.show()

#HISTOGRAMAS
np.histogram(red.PROMEDIOTempl, bins=[7,8,9,10,11,12,13,14,15])
print(np.histogram(white.PROMEDIOTemp, bins=[7,8,9,10,11,12,13,14,15]))