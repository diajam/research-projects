# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 11:11:20 2018

@author: James
"""
import analyse_f as ana
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def analyse_IV(t,i,v):
    plt.plot(t,v)
    #plt.plot(t,d*1000)
    plt.show()
    
    # Graphique de la puissance P(t) = i(t)*v(t)
    p=ana.graph_P(i,v)
    
    # Démarche pour trouver la tension de claquage:
    # 1: on trouve le point où di/dt est le plus grand pour l'anode
    # et le point où di/dt est le plus petit pour la cathode
    s = np.zeros(len(i))
    r = np.zeros(len(i))
    for k in range(len(i)-1):
        if t[k] < 50e-6:
            s[k] = (i[k+1]-i[k])/(t[k+1]-t[k])
            r[k] = 0
        if t[k] > 50e-6 and t[k] < 65e-6:
            s[k] = 0
            r[k] = (i[k+1]-i[k])/(t[k+1]-t[k])
    print(max(s))
    z = np.where(s == max(s))
    y = np.where(r == min(r))
    
    #2: Acquisition des données
    t1=t[z[0][0]];t3=t[y[0][0]]
    i1=i[z[0][0]];i3=i[y[0][0]] 
    x=np.where((i3-i) > 0.00001)
    w=np.where((i-i1) > 0.00001)
    u=np.where(t[w[0]] < 4e-5)
    t4=t[x[0][-1]];i4=i[x[0][-1]]
    try:
        t2=t[w[0][u[0][-1]]]
        i2=i[w[0][u[0][-1]]]
    except IndexError:
        t2=t[0]
        i2=0.
    
    # Points de séparations pour les calculs de charge et de l'énergie 
    # dissipé
    # Calcul et acquisition de la charge et de l'énergie dissipée 
    # par la décharge
    Q=ana.densite_charge(i,t,50e-6)
    E=ana.densite_charge(p,t,50e-6)
    
    #Validation
    plt.plot(t1,i1,'o')
    plt.plot(t2,i2,'o')
    plt.plot(t3,i3,'o')
    plt.plot(t4,i4,'o')
    plt.plot(t,i)
    plt.show()
    
    return [Q,E,v[z[0][0]],abs(v[y[0][0]]),(t2-t1),(t4-t3)]
    
path='F:/24-01-22/'
dirs=os.listdir(path)
#print(dirs)
Q_list = [];Q_err_list= []
E_list = [];E_err_list= []

low_peak_list = [];high_peak_list=[]
for i in dirs:
    #print(i)
    path2 = os.path.join(path,str(i))
    #print(path2)
    
    dirs2 = os.listdir(path2)
    print(dirs2)
    
    Q=[];Q_err=[]
    E=[];E_err=[]
    
    for j in range(1,len(dirs2)+1):
        
        #print(j)
        path3 = os.path.join(path2,str(j))
        dirs3 = os.listdir(path3)
        l = 1
        q = []
        e = []
        
        for k in dirs3:
            
            #print(k)
            fichier = pd.read_csv(path3+'/'+k,skiprows = 10)
            
            #print(fichier['TIME'])
            
            t = fichier['TIME']
            v = fichier['CH1']
            c = fichier['CH2']
            
            p = v*c*2
            
            q.append(ana.densite_charge(c*2,t))
            e.append(ana.densite_charge(p,t))
            
            a= find_peaks(c*2,prominence=.25,distance=75)
            b = a[0][np.where(np.logical_and(t[a[0]]>.7e-7,t[a[0]]<5e-7))]
                     
            m = [];m_t=[]
            print(b)
            print('I am in danger')
            for i in range(0,len(b)-1):
                
                print(t[b[i]],t[b[i+1]])
                
                d = np.where(np.logical_and(t>t[b[i]],t<t[b[i+1]]))
                
                g = np.where(np.logical_and(c==np.min(c[d[0]]),np.logical_and(t>t[b[i]],t<t[b[i+1]])))
                
                m.append(c[g[0][0]]*2);m_t.append(t[g[0][0]])
                
                '''
                plt.plot(t,c)
                plt.plot(t[d[0]],c[d[0]])
                plt.plot(t[g[0]],c[g[0]],'o')
                plt.xlim([1e-7,5e-7])
                plt.show()
                
                #print(d)
                '''
                
            #clean up step
            
            C_peaks = np.asarray(c[a[0][np.where(np.logical_and(t[a[0]]>.7e-7,t[a[0]]<5e-7))]]*2)
            
            lower_peaks = [];lower_peaks_time =[]
            higher_peaks = [];higher_peaks_time=[]
            
            for i in range(0,len(m)):
            
                if m[i]>5:
                    lower_peaks.append(m[i])
                    lower_peaks_time.append(m_t[i])
                else:
                    pass
                
            for i in range(0,len(C_peaks)):
            
                if C_peaks[i]>5:
                    higher_peaks.append(C_peaks[i])
                    higher_peaks_time.append(t[b[i]])
                else:
                    pass
            
            
            print(m,C_peaks)
            
            fig, ax1=plt.subplots()

            ax1.plot(t,v,label='Tension')
            ax1.set_xlabel('Temps (s)')
            ax1.set_ylabel('Tension (V)')
            ax2=ax1.twinx()
            ax2.plot(t,c*2,color='r',label='Courant')
            ax2.plot(t[a[0]],c[a[0]]*2,'go')
            ax2.plot(m_t,m,'bo')
            ax2.set_ylabel('Courant (A)')
            ax2.set_ylim([6,16])
            plt.xlim([.5e-7,5e-7])
            
            #plt.savefig('../electro/data/106-150_1/'+str(j)+'/'+str(l)+'.jpg')
            plt.show()
            
            l+=1
            
            low_peak_list.append(np.asarray(lower_peaks));high_peak_list.append(np.asarray(higher_peaks))
        break
        Q.append(np.mean(q));Q_err.append(np.std(q))
        E.append(np.mean(e));E_err.append(np.std(e))
        
        
    break
    Q_list.append(Q);Q_err_list.append(Q_err)
    E_list.append(E);E_err_list.append(E_err)
