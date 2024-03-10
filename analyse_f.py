# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:31:09 2018

@author: James

This code consist of a toolbox of various fonctions used in other codes in this repository.

"""
import csv
import numpy as np
import matplotlib.pyplot as plt

def soustraction(i,j,k,l,m):
    
    #Ouverture des fichiers et acquisition des données
    t=[];t2=[];t7=[];t8=[];c=[];c2=[];v=[];v1=[]
    with open(i,newline='') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=',',quotechar='|')
        for row in spamreader:
            t.append(float(row[3])*1e6)
            c.append(-float(row[4])*10)

    with open(j,newline='') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=',',quotechar='|')
        for row in spamreader:
            t2.append(float(row[3])*1e6)
            c2.append(-float(row[4])*10)
    
    with open(k,newline='') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=',',quotechar='|')
        for row in spamreader:
            t7.append(float(row[3])*1e6)
            v.append(float(row[4]))

    with open(l,newline='') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=',',quotechar='|')
        for row in spamreader:
            t8.append(float(row[3])*1e6)
            v1.append(float(row[4])/1000)
    
    
    #Alignement temporel des deux graphes
    a=np.where(np.asarray(c)==max(c))
    b=t[a[0][0]]
    e=np.where(np.asarray(c2)==min(c2))
    f=t2[e[0][0]]
    dt=b-f
    print (dt)
    t1=[]
    #print(len(t1),len(t2))
    for i in range(len(t)):
        t1.append(t[i]-dt)
    
    #Réduction des données pour que les deux graphes commencent et
    #terminent aux mêmes endroits.
    #t3=[];t4=[];c3=[];c4=[]
    for i in range(len(t1)):
        if t1[i]<t2[0]:
            t1.append(t1[i]+100)
            t1.remove(t1[i])
            c.append(c[i])
            c.remove(c[i])
    #print (t4[0],t4[-1],t3[0],t3[-1])
    #Soustraction de la mesure avec plasma au graphe sans plasma
    #et acquisition des données.
    di=[]
    dt=[]
    for i in range(len(t1)):
        try:
            s=np.where(np.asarray(t2)==round(t1[i],1))
            #print(i,t4[s[0][0]],t3[i])
            c_i=c2[s[0][0]]
            di.append(c_i-c[i])
            dt.append(t1[i])
        except IndexError:
            pass
    #print(di,dt)
    #plt.plot(t1,c,label='Reference')
    #plt.plot(t2,c2,label='Mesure')
    #plt.plot(t2,c2,label='Mesure')
    #plt.plot(dt,di,label='Mesure - Reference')
    #plt.xlim([t2[0],t2[-1]])
    #plt.savefig(m+'.jpg')
    #plt.legend()
    plt.show()
    plt.plot(t7,v,label='tension référence')
    plt.plot(t,c,label='courant référence')
    plt.plot(t2,c2,label='courant mesure')
    #plt.plot(t1,v)
    plt.plot(t8,v1,label='tension mesure')
    plt.legend()
    plt.savefig(m+'.jpg')
    plt.show()
    return np.array([di,dt])

def p_moy(p,t,td):
    p_i=0
    for j in range(len(t)-1):
        p_i+= (t[j+1]-t[j])*(p[j]+p[j+1])/2
    p_m=p_i/td
    return p_m
def calc_m_std(a,b,c):
    print(len(a))
    t_m=np.zeros(len(a))
    t_err=np.zeros(len(a))
    for i in range(len(a)):
        try:
        #print([a[i],b[i],c[i]])
            t=np.array([a[i],b[i],c[i]])
            t_m[i]=np.mean(t)
            t_err[i]=np.std(t)
        except IndexError:
            try:
                print([a[i],b[i]])
                t=np.array([a[i],b[i]])
                t_m[i]=np.mean(t)
                t_err[i]=np.std(t)
            except IndexError:
                t=np.array([a[i]])
                t_m[i]=np.mean(t)
                t_err[i]=np.std(t)
    return [t_m,t_err]

def densite_charge(i,t):
    q1=0
    for j in range(len(t)-1):
        A = abs((t[j+1]-t[j])*(i[j]+i[j+1])/2)
        if A == inf or A == NaN:
            pass
        else:
            q1+=A
                #print('t est entre ', t1,' et ', t2)
    return q1

def graph_raw_data(i,j,k):
    with open(i,newline='') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=',',quotechar='|')
        t=[]
        v=[]
        for row in spamreader:
            t.append(row[3])
            v.append(row[4])
        
    with open(j,newline='') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=',',quotechar='|')
        t1=[]
        i=[]
        for row in spamreader:
            t1.append(row[3])
            i.append(row[4])
        
    fig, ax1=plt.subplots()

    ax1.plot(t,v,label='Tension')
    ax1.set_xlabel('Temps (s)')
    ax1.set_ylabel('Tension (V)')
    ax2=ax1.twinx()
    ax2.plot(t1,i,color='r',label='Courant')
    ax2.set_ylabel('Courant (A)')
    ax2.plot(np.nan, label = 'Tension')
    ax2.legend(loc=1)
    plt.savefig(k+'.jpg')
    return

def graph_v(i,d):
    with open(i,newline='') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=',',quotechar='|')
        t=[]
        v=[]
        for row in spamreader:
            t.append(float(row[3]))
            v.append(float(row[4])*1000)
    t1=[]
    v1=[]
    for i in range(len(t)):
       if t[i]>-100e-6 and t[i]<0e-6:
            t1.append(t[i])
            v1.append(v[i])
    plt.plot(t1,v1,'o',label='d= '+str(d))
    return v1,t1

def graph_c(i,d):
    with open(i,newline='') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=',',quotechar='|')
        t=[]
        c=[]
        for row in spamreader:
            t.append(float(row[3]))
            c.append(float(row[4]))
    t1=[]
    c1=[]
    for i in range(len(t)):
        if t[i]>-100e-6 and t[i]<0e-6:
            t1.append(t[i])
            c1.append(c[i])
    plt.plot(t1,c1,'o',label='d= '+str(d))
    return c1

def graph_P(i,v):
    p=np.zeros(len(v))
    for j in range(len(v)):
        p[j]=(abs(float(v[j]))*abs(float(i[j])))
    
    #plt.plot(t,p,label='d= '+str(d))
    return p

def graph_R(i,j,d):
    with open(i,'rb') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=',',quotechar='|')
        t=[]
        v=[]
        for row in spamreader:
            t.append(float(row[3])*1e6)
            v.append(row[4])
        
    with open(j,'rb') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=',',quotechar='|')
        t1=[]
        i=[]
        for row in spamreader:
            t1.append(row[3])
            i.append(row[4])
    r=[]
    for j in range(len(v)):
        r.append(abs(float(v[j]))/abs(float(i[j])))
    
    plt.plot(t,r,label='d= '+str(d))
    return
#s1=soustraction('ref\\refsinus1000_Ch2.cs','V_1050\\16_35mm\\ALL0002\\F0002CH2.CSV','ref\\refsinus1000_Ch1.csv','V_1050\\16_35mm\\ALL0002\\F0002CH1.CSV','1')
#s1=soustraction('ref\\refsinus1000_Ch2.csv','V_1050\\16_35mm\\ALL0003\\F0003CH2.CSV','ref\\refsinus1000_Ch1.csv','V_1050\\16_35mm\\ALL0003\\F0003CH1.CSV','1')
#path=input('Name of the file: ')
#dirs=os.listdir(path)

def Lorentzian(x1,amp1,wid1,x0):
    return amp1*wid1**2/((x1-x0)**2+wid1**2)

def Gaussian(x1,amp1,wid1,x0):
    return amp1*np.exp(-((x1-x0)**2)/(2*wid1**2))

def Voigt(x, ampG1, cenG1, sigmaG1, ampL1, cenL1, widL1):

    return (ampG1*(1/(sigmaG1*(np.sqrt(2*np.pi))))*(np.exp(-((x-cenG1)**2)/((2*sigmaG1)**2)))) +\
            ((ampL1*widL1**2/((x-cenL1)**2+widL1**2)) )
