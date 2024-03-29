'''

This code was used to do signal analysis. Using current and voltage curves from an oscilloscope, I calculate different values by importing functions from another file
named analyse_f.

At the end, I create a csv file with the relevant data for other uses.
'''

import analyse_f as ana
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm.notebook import tqdm, trange


Q_list = [];Q_err_list= []
E_list = [];E_err_list= []
S_list = []
for i in dirs:
    
    
    print(i)
    path2 = os.path.join(path,str(i))
    #print(path2)
    
    dirs2 = os.listdir(path2)
    #print(dirs2)
    
    Q=[];Q_err=[]
    E=[];E_err=[]
    S=[]
    
    pbar = tqdm(total = len(dirs2))
    for j in range(1,len(dirs2)+1):
        
        
        
        low_peak_list = [];high_peak_list=[]
        low_peak_time_list = [];high_peak_time_list = []
        #print(j)
        path3 = os.path.join(path2,str(j))
        dirs3 = os.listdir(path3)
        l = 1
        q = []
        e = []
        
        s = 0 #Nombre de décharge réussie
        
       
        
        for k in dirs3:
            
            #print(k)
            fichier = pd.read_csv(path3+'/'+k,skiprows = 10)
            
            #print(fichier['TIME'])
            
            t = fichier['TIME']
            v = fichier['CH1']
            c = fichier['CH2']
            
            p = abs(v*c*2)
            
            if max(c*2 > 4):
                s += 1
                q.append(np.trapz(c[np.where(c<1e78)[0]]*2,t[np.where(c<1e78)[0]]))
                e.append(np.trapz(p[np.where(p<1e78)[0]],t[np.where(p<1e78)[0]]))
                
            
            a= find_peaks(c*2,prominence=.25,distance=75)
            b = a[0][np.where(np.logical_and(t[a[0]]>.7e-7,t[a[0]]<5e-7))]
                     
            m = [];m_t=[]
            
            for l in range(0,len(b)-1):
                
                d = np.where(np.logical_and(t>t[b[l]],t<t[b[l+1]]))
                
                g = np.where(np.logical_and(c==np.min(c[d[0]]),np.logical_and(t>t[b[l]],t<t[b[l+1]])))
                
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
            
            for n in range(0,len(m)):
            
                if m[n]>5:
                    lower_peaks.append(m[n])
                    lower_peaks_time.append(m_t[n])
                else:
                    pass
                
            for n in range(0,len(C_peaks)):
            
                if C_peaks[n]>5:
                    higher_peaks.append(C_peaks[n])
                    higher_peaks_time.append(t[b[n]])
                else:
                    pass
            
            low_peak_list.append(np.asarray(lower_peaks));high_peak_list.append(np.asarray(higher_peaks))
            low_peak_time_list.append(np.array(lower_peaks_time));high_peak_time_list.append(np.array(higher_peaks_time))
            
            
        #save_peaks(i,j,high_peak_list,high_peak_time_list,low_peak_list,low_peak_time_list)
        pbar.update(1)
        Q.append(np.mean(q));Q_err.append(np.std(q))
        E.append(np.mean(e));E_err.append(np.std(e))
        S.append(s)
        
        
        #print('minute ',j,' finished!')
    pbar.close() 
    t = np.arange(1,21)
    Q_list.append(Q);Q_err_list.append(Q_err)
    E_list.append(E);E_err_list.append(E_err)
    S_list.append(S)
    save_qe('106-150',i,t,Q,Q_err,E,E_err,S)

def save_qe(taille,distance,t,q,q_err,e,e_err,s):
    
    d1 = {'time (min)':t,'Charge (C)':q,'Charge error':q_err,'Energy (J)':e,'Energy error':e_err,'Probability':s}
    df1 = pd.DataFrame(data=d1)
    
    try:
        df1.to_csv('Results/electro2/'+taille+'_'+distance+'.txt', index=None,sep='\t',mode='w')
       
    except OSError:
        os.mkdir('Results/electro2')
        df1.to_csv('Results/electro2/'+taille+'_'+distance+'.txt', index=None,sep='\t',mode='w')
    return
