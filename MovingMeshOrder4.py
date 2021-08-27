#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 18:50:32 2021

@author: rmcclarr
"""


from numba import njit, prange, uintc, float64, int32, int64
import numpy as np
import matplotlib.pyplot as plt
import math
import quadpy
from scipy.integrate import solve_ivp
#need to fix the evaluation of position
@njit("float64[:](float64,float64[:],float64[:],float64[:],float64[:],int64,float64[:,:,:],float64)", looplift=True, parallel=True)
def rhs(t,mu,w, sigmat, sigmas, I, oldval,L):
    hx = L/I
    Q = lambda a,b,t: ((math.exp(-t)/(2*t+L)*math.sqrt(b-a),0,0,0,0))
#    at = []
#    bt = []
#    for i in range(I//2):
#        at.append(lambda i,t: left[i] + 1*t*np.min(mu)*left[i]/left[0])
#        bt.append(lambda i,t: right[i] + 1*t*np.min(mu)*right[i]/left[0])
#    for i in range(I//2,I):
#        at.append(lambda i,t: left[i] + 1*t*np.max(mu)*left[i]/right[I-1])
#        bt.append(lambda i,t: right[i] + 1*t*np.max(mu)*right[i]/right[I-1])
    K = int(mu.size)
    left = np.linspace(-L/2,L/2-hx,I)
    right = np.linspace(-L/2+hx,L/2,I)
    #tmp = np.array(oldval)
    psi = oldval.copy() #reshape((I,K,5))
    phi = np.zeros((I,5))
    phi[:,0] = (psi[:,:,0]*w).sum(axis=1)
    phi[:,1] = (psi[:,:,1]*w).sum(axis=1)
    phi[:,2] = (psi[:,:,2]*w).sum(axis=1)
    phi[:,3] = (psi[:,:,3]*w).sum(axis=1)
    phi[:,4] = (psi[:,:,4]*w).sum(axis=1)
    sqrt3 = 1.7320508075688772 # sqrt 3
    sqrt5 = 2.23606797749979 # sqrt 5
    sqrt7 = 2.6457513110645907 # sqrt 7
    RHS = np.zeros((I,K,5))
    MK = np.array(((0,0,0,0,0),
                   (2*sqrt3,0,0,0,0),
                   (0,2*sqrt3*sqrt5,0,0,0),
                   (2*sqrt7,0,2*sqrt5*sqrt7,0,0),
                   (0,6*sqrt3,0,6*sqrt7,0)))
    
    for i in prange(I):
        atval = left[i] + 1*t*np.min(mu)*left[i]/left[0] #bt[i](i,t)
        a = 1*np.min(mu)*left[i]/left[0]
        if (i>= I//2):
            atval = left[i] + 1*t*np.max(mu)*left[i]/right[I-1]
            a = 1*np.max(mu)*left[i]/right[I-1]
        #atval = at[i](i,t)
        btval = right[i] + 1*t*np.min(mu)*right[i]/left[0]
        b = 1*np.min(mu)*right[i]/left[0]
        if (i>=I//2):
            btval = right[i] + 1*t*np.max(mu)*right[i]/right[I-1]
            b = np.max(mu)*right[i]/right[I-1]
        #xmid = (bt[i](i,t) + at[i](i,t))*0.5
        h = btval - atval
        ih = 1/h
        #b = (bt[i](i,t+delta*1j)).imag/delta
        #a = (at[i](i,t+delta*1j)).imag/delta
        MG = np.array(((-0.5*(b-a)*ih,0,0,0,0),
                       (-sqrt3*ih*(b+a),-1.5*ih*(b-a),0,0,0),
                       (-sqrt5*(b-a)*ih,-sqrt5*sqrt3*(b+a)*ih,2.5*(a-b)*ih,0,0),
                       (-sqrt7*(b+a)*ih,-sqrt3*sqrt7*(b-a)*ih,-sqrt5*sqrt7*(a+b)*ih,3.5*ih*(a-b),0),
                       (-3*ih*(b-a),-3*sqrt3*(b+a)*ih,-3*sqrt5*(b-a)*ih,-3*sqrt7*ih*(b+a),-4.5*ih*(b-a))))
        
        Qtmp = np.array(Q(atval,btval,t))
        for k in range(K):
            right_speed = mu[k] - b
            left_speed = mu[k] - a
            psiR = 0.0
            psiL = 0.0
            if (right_speed >= 0):
                psiR = ih*right_speed*(psi[i,k,0]+sqrt3*psi[i,k,1]+sqrt5*psi[i,k,2]+sqrt7*psi[i,k,3]+3*psi[i,k,4])
            elif (i<I-1):
                psiR = ih*right_speed*(psi[i+1,k,0]-sqrt3*psi[i+1,k,1]+sqrt5*psi[i+1,k,2]-sqrt7*psi[i+1,k,3]+3*psi[i+1,k,4])
            if (left_speed >= 0) and (i>0):
                psiL = ih*left_speed*(psi[i-1,k,0] + sqrt3*psi[i-1,k,1]+sqrt5*psi[i-1,k,2]+sqrt7*psi[i-1,k,3]+3*psi[i-1,k,4])
            elif (left_speed < 0):
                psiL = ih*left_speed*(psi[i,k,0] - sqrt3*psi[i,k,1]+sqrt5*psi[i,k,2]-sqrt7*psi[i,k,3]+3*psi[i,k,4])

                
            RHS[i,k,:] -= np.array(((psiR - psiL),sqrt3*(psiR + psiL),sqrt5*(psiR - psiL),sqrt7*(psiR + psiL),3*(psiR - psiL)))
      #      RHS[i,k,1] -= sqrt3*ih*(psiR + psiL)
      #      RHS[i,k,2] -= sqrt5*ih*(psiR - psiL)
      #      RHS[i,k,3] -= sqrt7*ih*(psiR + psiL)
      #      RHS[i,k,4] -= 3*ih*(psiR - psiL)
            
            #total x-section and in-cell gradient/moving term
            RHS[i,k,:] -= sigmat[i]*psi[i,k,:] - np.dot(MG+mu[k]*MK*ih,psi[i,k,:])
            #source
            RHS[i,k,:] += phi[i,:]*sigmas[i] +  Qtmp
    return np.ravel(RHS) #RHS.reshape(I*K*5)
def make_output(sol,I,lt,rt,pts=5):
    h = rt-lt
    output = np.zeros(I*pts)
    points = np.zeros(I*pts)
    #x, w = np.polynomial.legendre.leggauss(pts)
    scheme = quadpy.c1.gauss_lobatto(pts)
    x = scheme.points
    #x = np.linspace(-1,1,pts)
    count = 0
    for i in range(I):
        for j in range(pts):
            output[count] = np.polynomial.legendre.legval(x[j],sol[i]*np.array((1,math.sqrt(3),math.sqrt(5),math.sqrt(7),math.sqrt(9))))/math.sqrt(h[i])
            points[count] = lt[i] + (x[j]+1)/(2)*h[i]
            count += 1
    return points, output
def planeSource(I = 64, L = 1e-6, Nang = 18, tf = 1):
    hx = L/I
    left = np.linspace(-L/2,L/2-hx,I)
    right = np.linspace(-L/2+hx,L/2,I)
    at = []
    bt = []
    for i in range(I//2):
        at.append(lambda i,t: left[i] + 1*t*np.min(mu)*left[i]/left[0])
        bt.append(lambda i,t: right[i] + 1*t*np.min(mu)*right[i]/left[0])
    for i in range(I//2,I):
        at.append(lambda i,t: left[i] + 1*t*np.max(mu)*left[i]/right[I-1])
        bt.append(lambda i,t: right[i] + 1*t*np.max(mu)*right[i]/right[I-1])
    #mu, w = np.polynomial.legendre.leggauss(Nang)
    scheme = quadpy.c1.gauss_lobatto(Nang)
    mu = scheme.points
    w = scheme.weights
    w /= np.sum(w)
    #print("mu = %s,\nw = %s" %(mu,w))
    sigmat = np.zeros(I) + 1.0
    sigmas = np.zeros(I) + 1.0
    Q = lambda a,b,t: ((math.exp(-t)/(2*t+L)*math.sqrt(b-a),0,0,0,0))
    ode_rhs = lambda t,y: rhs(t, mu,w, sigmat,sigmas, I, y.reshape((I,Nang,5)),L)
    ic = np.zeros((I,mu.size,5))
    sol = solve_ivp(ode_rhs, [0, tf], ic.reshape(I*mu.size*5),rtol=1e-9,atol=1e-11, t_eval=[0,tf], method="DOP853")#, method="Radau", vectorized=False)
    phi = np.zeros((I,5))
    print("Solver termination reason:", sol.message)
    tind = -1
    oput = sol.y[:,tind].reshape((I,mu.size,5))
    lt = np.zeros(I)
    rt = np.zeros(I)
    tind = -1
    for i in range(I):
        lt[i] = at[i](i,sol.t[tind])
        rt[i] = bt[i](i,sol.t[tind])
    for j in range(5):
        phi[:,j] = np.sum(w*oput[:,:,j],axis=1)
    #phi[:,0] += Q(lt[0],rt[-1],tf)[0]/np.sqrt(rt[-1]-lt[0])
    return lt, rt, phi


#test planeSource
Is =  np.array([2,4,8,16,32,64,128,256]) #np.array([128]) #
errs = 0.0*Is
errRMS = 0.0*Is
i = 0
from scipy.interpolate import interp1d
pl = np.loadtxt("plane001.dat")
ganapol = interp1d(pl[:,0],pl[:,1], kind="cubic") 
for I in Is:
    L = 1e-12
    lt, rt, phi = planeSource(I, L, Nang = 2*2048)
    points, output = make_output(phi, I, lt, rt)
    plt.plot(points, output+np.exp(-1)/(2+L),'o--')
    plt.plot(pl[:,0], pl[:,1],'k'); plt.plot(-pl[:,0], pl[:,1], 'k')
    plt.show()
    sel = np.argmin(np.abs(0.5*(lt+rt)-pl[50,0]))
    nval = phi[sel,0]/np.sqrt(rt[sel]-lt[sel])+np.exp(-1)/(2+L)
    print(points[points.size//2],output[points.size//2]+np.exp(-1)/(2+L))
    print(pl[0,:])
    print(output[[points.size//2]]+np.exp(-1)/(2+L) - pl[0,1])
    errs[i] = nval-ganapol(0.5*(lt+rt)[sel])
    errRMS[i] = np.sqrt(np.mean((output+np.exp(-1)/(2+L) - ganapol(np.abs(points)))**2))
    print("I =", I,"RMS =",errRMS[i])
    i += 1
    np.savetxt("rmserror.npy",errRMS)