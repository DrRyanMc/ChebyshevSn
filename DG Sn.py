# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 09:28:07 2021


"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 10:17:08 2021

@author: williambennett
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import numpy.polynomial.chebyshev as C
import numpy as np
import matplotlib.pyplot as plt
#from scipy.interpolate import lagrange
from scipy.integrate import solve_ivp
import scipy.integrate as integrate
#from scipy.special import roots_legendre
from numba import jit, uintc, float64,njit
import quadpy
#from decimal import *
from scipy.special import eval_legendre, lpmv
import math
# @jit(float64(float64,float64,float64))
def Bi_func(x,i,xL,xR):
    # orthonormal basis functions 
#    return (math.sqrt(1 + 2*i)*eval_legendre(i,(-2*x + xL + xR)/(xL - xR)))/math.sqrt(-xL + xR)
    return (math.sqrt(1 + 2*i)*lpmv(0,i,(-2*x + xL + xR)/(xL - xR)))/math.sqrt(-xL + xR)
def Bi_func2(x,i,xL,xR):
    # orthonormal basis functions squared
    return ((math.sqrt(1 + 2*i)/math.sqrt(xR-xL)*(eval_legendre(i,(-2*x + xL + xR)/(xL - xR)))))**2
def BjdBidx_func(x,i,j,t,N_pnts,xL,xR):
    z = (xL+ xR-2*x)/(xL-xR)
    Bj = np.sqrt(2*j+1)/np.sqrt((xR-xL))*eval_legendre(j,z)
    dBidx = dBdx_func(i,x,z,t,xL,xR)
    return Bj*dBidx
def BjBi_func(x,i,j,t,N_pnts,xL,xR):
    z = (xL+ xR-2*x)/(xL-xR)
    Bj = np.sqrt(2*j+1)/np.sqrt((xR-xL))*eval_legendre(j,z)
    Bi = np.sqrt(2*i+1)/np.sqrt((xR-xL))*eval_legendre(i,z)
    return Bi*Bj
def dBdt_func(i,x,t,xL,xR,dxL,dxR):
    result = (np.sqrt(1 + 2*i)*(eval_legendre(i,(-2*x + xL + xR)/(xL - xR))*(dxL - dxR) + (4*(1 + i)*(-xL + xR)*(-(((-2*x + xL + xR)*eval_legendre(i,(-2*x + xL + xR)/(xL - xR)))/(xL - xR)) + eval_legendre(1 + i,(-2*x + xL + xR)/(xL - xR)))*((x - xR)*dxL + (-x + xL)*dxR))/((xL - xR)**2*(-1 + (-2*x + xL + xR)**2/(xL - xR)**2))))/(2.*(-xL + xR)**1.5)
    return result 
def dBdx_func(i,x,z,t,xL,xR):
    dx = xR-xL
    term1 = (i+1)*np.sqrt(2*i+1)*np.sqrt(dx)
    term2 = -eval_legendre(i,z)*z+eval_legendre(i+1,z)
    term3 = 2*(x-xL)*(x-xR)
    result = term1*(term2)/term3
    return result
def Bi_phi(x,i,tt,xL,xR):
    t = tt + 1e-12
    return Bi_func(x,i,xL,xR)*np.exp(-t)*np.heaviside((t+x)/t,0)*np.heaviside(1-x/t,0)/(2*t)
def phi_u(x,tt):
    t = tt 
    return np.exp(-t)*np.heaviside((t+x)/t,0)*np.heaviside(1-x/t,0)/(2*t)
@jit
def grid_func(k,N_space,t,left,right,dx,mus):
    if (k < N_space//2):
        xL = left[k] + 1*t*np.min(mus)*left[k]/left[0] #bt[i](i,t)
        dxL = 1*np.min(mus)*left[k]/left[0]
        xR = right[k] + 1*t*np.min(mus)*right[k]/left[0]
        dxR = 1*np.min(mus)*right[k]/left[0]
    elif (k>= N_space//2):
        xL = left[k] + 1*t*np.max(mus)*left[k]/right[N_space-1]
        dxL = 1*np.max(mus)*left[k]/right[N_space-1]
        xR = right[k] + 1*t*np.max(mus)*right[k]/right[N_space-1]
        dxR = np.max(mus)*right[k]/right[N_space-1]
    center = (xL+xR)/2
    return xL,xR,dxL,dxR,center
@jit
def G_func(k,t,N_space,M,dx,left,right,xL,xR,dxL,dxR):
    h = xR - xL
    ih = 1/h
    b = dxR
    a = dxL
    G = np.zeros((M+1,M+1))
    for i in range(0,M+1):
        for j in range(0,M+1):
            ##############################################################
            # if i>=j:
            #     G[i,j] = integrate.quad(BjdBidt_func,xL,xR,args=(i,j,k,t,xL,xR,dxL,dxR))[0]
            ##############################################################
            if i==j:
                G[i,j] = -0.5*(2*i+1)*ih*(b-a)
            elif i>j:
                if (i+j)%2 ==0:
                    G[i,j] = -math.sqrt(2*j+1)*math.sqrt(2*i+1)*ih*(b-a)
                else:
                    G[i,j] = -math.sqrt(2*j+1)*math.sqrt(2*i+1)*ih*(b+a)
    return G
def L_func(t,N_pnts,M,xL,xR):
    L = np.zeros((M+1,M+1))
    for i in range(0,M+1):
        for j in range(0,M+1):
            if i > j and (i+j) % 2 !=0: 
                L[i,j] = integrate.quad(BjdBidx_func,xL,xR,args=(i,j,t,N_pnts,xL,xR))[0]
            else:
                L[i,j] = 0
    return L 
@jit
def LU_surf_func(u,space,N_space,mul,M,xL,xR,dxL,dxR):
    sumright = 0
    sumleft = 0
    rightspeed = mul - dxR
    leftspeed = mul-dxL
    for j in range(0,M+1):
        sumright += surf_func(rightspeed,u,space,j,"R",xL,xR,N_space)
        sumleft += surf_func(leftspeed,u,space,j,"L",xL,xR,N_space)
    LU = np.zeros(M+1).transpose()
    for i in range(0,M+1):
        if i == 0:
            B_right = 1/math.sqrt(xR-xL)
            B_left = 1/math.sqrt(xR-xL)
        elif j>0:
            B_right = math.sqrt(2*i+1)/math.sqrt(xR-xL)
            if i%2 ==0:
                B_left = math.sqrt(2*i+1)/math.sqrt(xR-xL)
            else: 
                B_left = -math.sqrt(2*i+1)/math.sqrt(xR-xL)
        LU[i] = rightspeed*B_right*(sumright) - leftspeed*B_left*(sumleft)
    return LU 
@jit 
def surf_func(speed,u,space,j,side,xL,xR,N_space):
#    print(side)
    if j ==0:
        B_right = 1/math.sqrt(xR-xL)
        B_left = 1/math.sqrt(xR-xL)
    else:
         B_right = math.sqrt(2*j+1)/math.sqrt(xR-xL)
         if j%2 ==0:
                B_left = math.sqrt(2*j+1)/math.sqrt(xR-xL)
         else:
                B_left = -math.sqrt(2*j+1)/math.sqrt(xR-xL)
    if speed == 0:
        return 0
    elif speed > 0 and side == "R":
        return u[space,j]*B_right
    elif speed > 0 and side =="L":
        if space !=0:
#            print(u[k-1,j])
            return u[space-1,j]*B_right 
        else:
            return 0
    elif speed < 0 and side =="R":
        if space != N_space-1:
            return u[space+1,j]*B_left
        else:
            return 0
    elif speed < 0 and side =="L":
        return u[space,j]*B_left
def P_func(t,M,xL,xR):
    P = np.zeros(M+1).transpose()
    if t>0:
        if ((xR<=t) and (xL>=-t)):
            P[0] = math.exp(-t)*math.sqrt(xR-xL)/2/t
        elif (t <= xL and t<xR) or (-t >= xL and -t>xR) :
            return P
        else:
            for i in range(0,M+1):
                    if (xR>t and xL>-t):
                        if i ==0:
                            result = (t - xL)/(2.*math.exp(t)*t*math.sqrt(-xL + xR))
                        else:
    #                        result = integrate.quad(Bi_func,xL,t,args=(i,xL,xR))[0]*math.exp(-t)/2/t
                            result = ((xL - xR)*(eval_legendre(-1 + i,(-2*t + xL + xR)/(xL - xR)) - eval_legendre(1 + i,(-2*t + xL + xR)/(xL - xR))))/(4.*math.exp(t)*t*math.sqrt((1 + 2*i)*(-xL + xR)))
                    elif (xR>t and xL <-t):
                        if i ==0:
                            result = 1/(math.exp(t)*math.sqrt(-xL + xR))
                        else:
                            result = ((xL - xR)*(eval_legendre(-1 + i,(-2*t + xL + xR)/(xL - xR)) - eval_legendre(-1 + i,(2*t + xL + xR)/(xL - xR)) - eval_legendre(1 + i,(-2*t + xL + xR)/(xL - xR)) + eval_legendre(1 + i,(2*t + xL + xR)/(xL - xR))))/(4.*math.exp(t)*t*math.sqrt((1 + 2*i)*(-xL + xR)))
                    elif (xR<t and xL<-t):
                        if i ==0:
                            result = (t + xR)/(2.*math.exp(t)*t*math.sqrt(-xL + xR))
                        else:
    #                        result = integrate.quad(Bi_func,-t,xR,args=(i,xL,xR))[0]*math.exp(-t)/2/t
                            result = -(math.sqrt(-(((1 + 2*i)*(t + xL)*(t + xR))/(xL - xR)**2))*math.sqrt(-xL + xR)*lpmv(1,i,(2*t + xL + xR)/(xL - xR)))/(2.*math.exp(t)*i*(1 + i)*t)
                    else:
    #                    print("zone integrated",xL,"left",xR,"right",t,"t")
                        result = integrate.quad(Bi_phi,xL,xR,args=(i,t,xL,xR))[0]
                    P[i] = result
    return P
def phi_sol_func(t,u,N_space,N_ang,M,ws,mus,dx,left,right):
    div = 100    
    t+=1e-12                 #number of subdivisions in each zone 
    xs_list = np.zeros((N_space*div))
    phi_list = np.zeros((N_space*div))
    j_list = np.zeros((N_space*div))
    for k in range(0,N_space):
        result = grid_func(k,N_space,t,left,right,dx,mus)
        xL = result[0]
        xR= result[1]
        xs = np.linspace(xL,xR,div)
        xs_list[k*div:(k+1)*div] = xs 
        sol_vec = np.zeros((N_ang,M+1,div))
        for ang in range(N_ang):
            for j in range(0,M+1):
                for ix in range(0,div):
                    sol_vec[ang,j,ix] = Bi_func(xs[ix],j,xL,xR)*u[ang,k,j]
        psi_c = np.zeros((N_ang,div)) 
        for ang2 in range(0,N_ang):
            psi_c[ang2,:] = np.sum(sol_vec[ang2,:,:],axis=0)
        phi_c = np.sum(np.multiply(psi_c.transpose(),ws),axis=1) 
        phi = phi_c + phi_u(xs,t+dx)
        j = np.sum(np.multiply(psi_c.transpose(),ws*mus),axis=1)/phi_c
#        phi_list[k*div:(k+1)*div] = phi
#        print(phi[0],phi[-1])
        phi_list[k*div+1:(k+1)*div-1] = phi[1:-1] 
        phi_list[k*div] += phi[0]
        phi_list[(k+1)*div-1] += phi[-1]
        j_list[k*div+1:(k+1)*div-1] = j[1:-1] 
        j_list[k*div] += j[0]
        j_list[(k+1)*div-1] += j[-1] 
    return xs_list,phi_list,j_list  
def nodevals(t,N_ang,N_space,M,u,ws,mus,dx,left,right):
    psi_c = np.zeros((N_ang,N_space))
    center_list = np.zeros(N_space)
    for k in range(N_space):
        result = grid_func(k,N_space,t,left,right,dx,mus)
        xL = result[0]
        xR= result[1]
        center = result[4]
        center_list[k] = center
        for ang in range(N_ang):
            for j in range(M+1):
                psi_c[ang,k] += Bi_func(center,j,xL,xR)*u[ang,k,j]
    phi_c = np.sum(np.multiply(psi_c.transpose(),ws),axis=1) 
    return center_list,phi_c
@jit
def isotropic_DG_split_rhs(t,V,N_space,N_ang,mus,ws,LL,M,sigma_t,sigma_s,dx,left,right,problem):
    """ Solves the equation:
         \frac{1}{c}\frac{\partial}{\partial t}\psi + \mu {\partial}{\partial x}\psi + \sigma_t \psi = \sigma_s \phi + S 
         By splitting into collided and uncollided (which will have a closed form solution) parts,
         Uncollided ->\frac{1}{c}\pD{}{t}\psi_u + \mu\pD{}{x}\psi_u + \sigma_t \psi_u =  S
         collided -> \frac{1}{c}\pD{}{t}\psi_c + \mu\pD{}{x}\psi_c + \sigma_t \psi_c = \sigma_s \phi
         (\pD{}{x_} = \frac{\partial}{\partial x_})
         \psi = \psu_u+\psi_c
         \phi = \int^1_{-1}(\psi_u+\psi_c)d\mu
         With discrete ordinance and discontinuous Galerkin, the collided equation becomes,
         $\frac{1}{c}\D{\boldsymbol{U}_l}{t}-\frac{1}{c}\left(\boldsymbol{GU}\right)_i^{\mathrm{(surf)}} - \frac{1}{c}\boldsymbol{GU}_l+ \mu_l\left(\boldsymbol{LU}_l\right)^{(\mathrm{surf})} - \mu_l\boldsymbol{LU}_l + \sigma_t \boldsymbol{U}_l =\sigma_s\sum_{m=1}^N
         w_m\boldsymbol{U}_m+\boldsymbol{P}$
         for every kth zone
         where,
         $\boldsymbol{\psi_c}^l(x,t) = \sum_k \sum_{j=0}^{M} B_{j}(z)\boldsymbol{\mathrm{u}}^l_{k,j}(t)$
         $u^l_k$ is the solution vector for angle l and zone k.
         $U_l$ is a vector of length M+1(number of basis functions) containing the $u^l_k's$
         The solution vector is dimension (N_ang,N_space,M+1)
    """
    V_new = V.copy().reshape((N_ang,N_space,M+1))
    V_old = V_new.copy()
    hx = dx/N_space
    left = np.linspace(-dx/2,dx/2-hx, N_space)
    right = np.linspace(-dx/2+hx,dx/2,N_space)
    for space in range(0,N_space):
        result = grid_func(space,N_space,t,left,right,dx,mus)
        xR= result[1]
        xL = result[0]
        dxL = result[2]
        dxR = result[3]
        phi_c = np.zeros(M+1).transpose()
        L = LL/(xR-xL)
        G = G_func(space,t,N_space,M,dx,left,right,xL,xR,dxL,dxR) 
        P = np.zeros(M+1).transpose()
        P[0] = (math.exp(-t+dx)/(2*t+dx)*math.sqrt(xR-xL))
        for i in range(0,M+1):
            phi_c[i]  = np.sum(np.multiply(V_old[:,space,i],ws))
        for angle in range(N_ang):
            mul = mus[angle]
            L_surf = LU_surf_func(V_old[angle,:,:],space,N_space,mul,M,xL,xR,dxL,dxR)
            U = np.zeros(M+1).transpose()
            U[:] = V_old[angle,space,:]
            ###################################################################
            RHS = np.dot(G,U) + -L_surf+ mul*np.dot(L,U) - sigma_t*U + sigma_s*phi_c + sigma_s*P
            V_new[angle,space,:] = RHS.transpose()
    return V_new.reshape(N_ang*N_space*(M+1))
def run_isotropic_DG(t=1,N_spaces=[2],N_angles=[256],Ms=[3],problem="ganapol"):
    sigma_s = 1
    sigma_t = 1
    dx = 1e-12
    for i in range(len(N_spaces)):
        N_space = N_spaces[i]
        N_ang = N_angles[i]
        hx = dx/N_space
        left = np.linspace(-dx/2,dx/2-hx, N_space)
        right = np.linspace(-dx/2+hx,dx/2,N_space)
        M = Ms[i]
        mus = quadpy.c1.gauss_lobatto(N_ang).points
        ws = quadpy.c1.gauss_lobatto(N_ang).weights
        ws = ws/np.sum(ws)
        IC = np.zeros((N_ang,N_space,M+1)) 
        L = L_func(0,N_space,M,-1/2,1/2) 
        rhs = lambda t,V: isotropic_DG_split_rhs(t,V,N_space,N_ang,mus,ws,L,M,sigma_t,sigma_s,dx,left,right,problem)
        sol = solve_ivp(rhs, [0.0,t], IC.reshape(N_ang*N_space*(M+1)), method='RK45')
        if not (sol.status == 0):
            print("solver failed %.0f"%N_space)
        sol_last = sol.y[:,-1].reshape((N_ang,N_space,M+1))
        xs,phi,j = phi_sol_func(t,sol_last,N_space,N_ang,M,ws,mus,dx,left,right) 
        center, nodes = nodevals(t,N_ang,N_space,M,sol_last,ws,mus,dx,left,right)
        plt.plot(xs,phi,"-")
        plt.scatter(center,nodes+phi_u(center,t),marker="x",label="M=%.0f N_space %.0f N_ang %.0f"%(M,N_space,N_ang))
        plt.plot(xs,j)
        plt.legend()
        plt.legend()  
        print(xs)
        if problem == "ganapol":
            if (t==1):
                sol = np.loadtxt("plane001.dat",delimiter="  ", usecols=[1,2])
                plt.plot(sol[:,0], sol[:,1],'k--')
                plt.plot(-sol[:,0], sol[:,1],'k--')
            if (t==5):
                sol = np.loadtxt("plane005.dat",delimiter="  ", usecols=[1,2])
                plt.plot(sol[:,0], sol[:,1],'k-')
                plt.plot(-sol[:,0], sol[:,1],'k-')
            if (t==10):
                sol = np.loadtxt("plane010.dat",delimiter="  ", usecols=[1,2])
                plt.plot(sol[:,0], sol[:,1],'k-')
                plt.plot(-sol[:,0], sol[:,1],'k-')
#        if problem =="steady":
#            plt.plot(xs, np.ones(len(xs))*1/(sigma_t-sigma_s))
        for k in range(0,N_space):
            plt.scatter(grid_func(k,N_space,t,left,right,dx,mus)[0],0,marker = "|",c="k")
            plt.scatter(grid_func(k,N_space,t,left,right,dx,mus)[1],0,marker = "|",c="k"),
    return sol_last 
ganapol = run_isotropic_DG(t=1,N_spaces=[32],N_angles=[64],Ms=[3],problem="ganapol")