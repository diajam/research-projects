'''
This code search and find the intensity of the higher peaks into the input data then compare them to the highest peak.
'''



import analyse_f as ana
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 
from scipy.signal import find_peaks

r_643_645 = []
for i in range(1,len(dirs1)+1):
    fichier = 'N3_656nm'+str(i)
    print(fichier)
    x,y = np.loadtxt(path+'/'+fichier+'.txt',delimiter='\t',unpack=True)
    
    x_bg,y_bg = np.loadtxt(path + '/../../bg2.txt',delimiter = '\t', unpack = True)
    
    y_m_bg= y-y_bg
    
    p = find_peaks(y_m_bg[np.where(np.logical_and(x<652,x>642))[0]],20000)
    #print(p)
    
    new_x = x[np.where(np.logical_and(x<652,x>642))[0]]
    new_y = y_m_bg[np.where(np.logical_and(x<652,x>642))[0]]
    
    y_n = new_y/new_y[p[0]][0]
    
    p2 = find_peaks(y_n,0.4,distance=1)
    
    plt.plot(new_x,y_n)
    plt.plot(new_x[p2[0]],y_n[p2[0]],'o')
    plt.xlim([642,652])
    plt.show()
    
    r_643_645.append(y_n[p2[0]][0]/y_n[p2[0]][1])
    
    print(new_x[p2[0]])

time = np.arange(1,len(dirs1)+1)
plt.plot(time,r_643_645,'o')
plt.show()
