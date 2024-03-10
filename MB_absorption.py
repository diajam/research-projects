import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Code simple permettant d'identifier l'intensité du pic 663 par absorption lumineuse d'une solution méthylène bleu en fonction du temps de traitement.

#Simple code to identify the intensity of the 663 peak by light absorption of a blue methylene solution as a function of treatment time.
#The code can be used in a loop over a directory where all the files for one treatment.


file = pd.read_csv('ar_100_o2_0.csv')

wavelenght = file['Baseline 100%T'][1:-1]
abs_30 = file['Unnamed: 11'][1:-1]

print(wavelenght.astype(int),abs_0.astype(float))

plt.plot(wavelenght.astype(int),abs_0.astype(float))
plt.show()

abs_0 = abs_0.astype(float)
a = np.where(abs_0 == np.max(abs_0))

x_max=a[0][0]

plt.plot(wavelenght.astype(int),abs_0.astype(float))
plt.plot(wavelenght.astype(int)[a[0][0]],abs_0.astype(float)[a[0][0]],'o')

plt.show()
