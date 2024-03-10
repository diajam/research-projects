# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 07:30:38 2018

@author: James
"""
import os
import pandas
import numpy as np
import matplotlib.colors as co
import matplotlib.cm as cm
import matplotlib.pyplot as plt


path = "Reacteur_2\\Images\\post_chute_-_2\\fichiers\\"

dirs = os.listdir(path)
P=[]
for file in dirs:

    f = open(path+file,mode='r')
    print (path+file)
    j = 0

    x=[]
    for line in f:
        x1 = line.split(',')
        x2=[]
        for i in x1:
            try:
                x2.append(float(i))
            except ValueError:
                pass
        del x2[0]
    
        x.append(x2)
    f.close()
    
    X=np.swapaxes(np.asarray(x),0,1)
    X2=pandas.DataFrame(X[10:255])
    X3=X2[X2.columns[495:545]]
    X4=np.asarray(X3)
    a=np.where(X4 == X4.max())
    
    print(a[0][0],a[1][0])
    pa = np.swapaxes(X4[0:245],0,1)
    P.append(pa[a[1][0]])
    x = np.linspace(0,245,245)
    b=np.where(np.log10(pa[a[1][0]]) >= 4.1)
    print(b[0])
    plt.plot(x,np.log10(pa[a[1][0]]))
    plt.plot(b[0],np.log10(pa[a[1][0]][b[0]]),'o')
    plt.show()
    pl = np.swapaxes(X4[b[0][0]-5:b[0][-1]+5],0,1)
    x2 = np.linspace(b[0][0],b[0][-1],(b[0][-1]-b[0][0])+10)
    PA = np.zeros(len(x2))
    PL = np.zeros(len(x2))
    ind =[]
    L=10000*np.ones(len(x2))
    for i in range (len(pl)):
        c=np.where(np.log10(pl[i]) >= 4.1)
        if len(c[0]) > 0:
            ind.append(i)
            for j in range(len(x2)):
                PA[j] += pl[i][j]
            
            #plt.plot(x2,np.log10(pl[i]),'o')
            #plt.plot(c[0],np.log10(pl[i][c[0]]),'o')
            #plt.show()
    pl2 = np.swapaxes(pl,0,1)
    for i in range(len(pl2)):
        d=np.where(np.log10(pl2[i]) >= 4.1)
        
        if len(d[0]) > 0:
            L[i]=len(d[0])
            #plt.plot(y,np.log10(pl2[i]))
            #plt.plot(ind[0],4.3,'o')
            #plt.plot(ind[-1],4.3,'o')
            #plt.show()
    
    plt.plot(x2,PA)
    plt.title('Profil axial (somme)')
    plt.show()
    
    plt.plot(x2,PA/L)
    plt.title('Profil axial (somme moyennee)')
    plt.show()
    
    plt.plot(x,pa[a[1][0]])
    plt.title('Profil axial (axe de l\'intensite max)')
    plt.show()
    
    fig,ax = plt.subplots()
    fig.set_figwidth(4)
    fig.set_figheight(5)
    im = ax.imshow(X4,cmap=cm.jet,norm=co.LogNorm(10e3,10e5),interpolation='gaussian') #,norm=co.LogNorm(X5.min(),X5.max())
    cbar = fig.colorbar(im,ax=ax)
    plt.ylim([0,244])
    #plt.xlim([375,535])
    plt.savefig("Reacteur_2\\Images\\post_chute_-_2\\"+file[:-4]+".png")
    plt.show()
    print (file)
