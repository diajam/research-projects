'''
This code take a spectrum taken from a spectrometer and try to fit experimental values to evaluate gauss, doppler and lorentz peak widths to a specific peak of a plasma discharge.
'''
import analyse_f as ana
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from scipy.signal import find_peaks

path = 'E:22-03-2022/20-38/distance1'
print(path.split('/'))
save_path = 'Results/Spectro2/'+path.split('/')[-2]+'/'+path.split('/')[-1]
dirs1 = os.listdir(path)
print(save_path)

#fit linéaire de la température en fonction du temps de traitement des poudres

lin = a*x + b

param = curve_fit(lin, x_a_fit, y_a_fit,np.array[100,7500])

def Voigt(x, ampL1, cenL1, widL1, sigmaG1):

    return (1*(1/(sigmaG1*(np.sqrt(2*np.pi))))*(np.exp(-((x-656.3)**2)/((2*sigmaG1)**2)))) +\
            ((ampL1*widL1**2/((x-656.3)**2+widL1**2)) )

L = []
L_err = []

L_V = []
L_V_err = []

L_G = []
L_G_err = []

for i in range(1,len(dirs1)+1):
    fichier = 'N3_656nm'+str(i)
    print(fichier)
    x,y = np.loadtxt(path+'/'+fichier+'.txt',delimiter='\t',unpack=True)
    
    x_bg,y_bg = np.loadtxt(path + '/../../bg2.txt',delimiter = '\t', unpack = True)
    
    y_m_bg= y-y_bg
    
    y_max = np.max(y_m_bg[np.where(np.logical_and(x>655,x<657))[0]])
    
    x_max = x[np.where(y_m_bg == y_max)[0]]
    
    Y = y_m_bg/y_max
    
    x_a_fit = x[np.where(np.logical_and(x>=652,x<=662))[0]]
    y_a_fit = Y[np.where(np.logical_and(x>=652,x<=662))[0]]
    
    a = np.where(y_a_fit<=np.max(y_a_fit/2))
    
    b = np.where(x_a_fit[a] < 656)
    c = np.where(x_a_fit[a] > 656)
    
    largeur = x_a_fit[a][c][0] - x_a_fit[a][b][-1]
    
    doppler = 7.16e-7 * 656.3 * np.sqrt(4500/1)
    
    experimental = 0.105
    
    gauss = np.sqrt(doppler**2+experimental**2)
    print('Largeurs [gauss, doppler, experimental]: ',gauss,', ', doppler,', ', experimental)
    
    try:
        params_V= curve_fit(lambda x_a_fit,ampL1,cenL1, widL1: Voigt(x_a_fit,1,656.3,widL1,gauss), x_a_fit, y_a_fit)
        
    except RuntimeError:
        pass
    
    Stark = params_V[0][2]
    Stark_err = np.sqrt(params_V[1][2][2])
    
    print('Largeurs [Lorentz]:', Stark)
    
    n_e = 1e20
    
    #fit = ana.Voigt(x_a_fit,params_V[0][0],params_V[0][1],params_V[0][2],params_V[0][3],params_V[0][4],params_V[0][5])
    
    plt.plot(x_a_fit,y_a_fit)
    plt.plot(x_a_fit,Voigt(x_a_fit,params_V[0][0],params_V[0][1],params_V[0][2],gauss))
    #plt.plot(x_a_fit,ana.Lorentzian(x_a_fit,params_L[0][0],params_L[0][1],params_L[0][2]))
    plt.plot(x_a_fit[a][b][-1],y_a_fit[a][b][-1],'ro')
    plt.plot(x_a_fit[a][c][0],y_a_fit[a][c][0],'go')
    #plt.ylim([0,1.1])
    plt.show()
    
time = np.arange(1,len(dirs1)+1)

print(len(time),len(L),len(L_V))

save_largeur(time,L_G,L_V,save_path)
save_courbes(x_a_fit,y_a_fit,fit,save_path)

plt.errorbar(time,L_G,yerr=L_G_err,fmt='o',label='Gauss')
plt.errorbar(time,L_V,yerr=L_V_err,fmt='o',label='Lorentz')
plt.legend()
plt.show()

def save_largeur(time,lorentz,voigt,path):
    
    d1 = {'Time (min)':time,'Largeur Gauss (nm)':lorentz,'Largeur Lorentz Voigt(nm)':voigt}
    

    df1 = pd.DataFrame(data=d1)
    
    df1.to_csv(path+'/Largeur.txt', index=None,sep='\t',mode='w')
    
    return

def save_courbes(wave,y_exp,y_theo,path):
    
    d1 = {'Wavelenght (nm)':wave, 'Relative Intensity':y_exp, 'Fit Voigt':fit}
    
    df1 = pd.DataFrame(data=d1)
    
    df1.to_csv(path+'/Courbes.txt', index=None, sep = '\t', mode = 'w')
    
    return




