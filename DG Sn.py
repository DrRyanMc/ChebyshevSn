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
from numba import jit, uintc, float64
import quadpy
#from decimal import *
from scipy.special import eval_legendre, lpmv
import math
@jit(float64(float64,float64,float64))
def legendre(j,i,x):
    return lpmv(j,i,x)
def Bi_func(x,i,xL,xR):
    # orthonormal basis functions 
#    return (math.sqrt(1 + 2*i)*eval_legendre(i,(-2*x + xL + xR)/(xL - xR)))/math.sqrt(-xL + xR)
    return (math.sqrt(1 + 2*i)*legendre(0,i,(-2*x + xL + xR)/(xL - xR)))/math.sqrt(-xL + xR)
def Bi_func2(x,i,xL,xR):
    # orthonormal basis functions squared
    return ((math.sqrt(1 + 2*i)/math.sqrt(xR-xL)*(eval_legendre(i,(-2*x + xL + xR)/(xL - xR)))))**2
def BjdBidt_func(x,i,j,k,t,N_pnts):
    # basis function B_j  X the time derivative of basis function B_i
    result = grid_func(k,N_pnts,t)
    xL = result[0]
    xR= result[1]
    dxL = result[2]
    dxR = result[3]
    z = (xL+ xR-2*x)/(xL-xR)
    Bj = np.sqrt(2*j+1)/np.sqrt((xR-xL))*eval_legendre(j,z)
    dBidt = dBdt_func(i,k,x,t,xL,xR,dxL,dxR)
    return Bj*dBidt
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
def dBdt_func(i,k,x,t,xL,xR,dxL,dxR):
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
def grid_func(k,N_pnts,t):
    L = -1.2
    R = 1.2
    D = R-L
    dx = D/N_pnts
    xL = L + k*dx
    xR = L+(k+1)*dx
    center = (xL+xR)/2
    ############################
    dxL = 0
    dxR = 0
    ############################
    return xL,xR,dxL,dxR,center
#for i in range(0,31):
#    print(grid_func(i+1,31,1)[0]-grid_func(i,31,1)[1])
#def G_func(k,t,N_pnts,M):
#    result = grid_func(k,N_pnts,t)
#    xL = result[0]
#    xR= result[1]
#    G = np.zeros((M+1,M+1))
#    for i in range(0,M+1):
#        for j in range(0,M+1):
#            G[i,j] = integrate.quad(BjdBidt_func,xL,xR,args=(i,j,k,t,N_pnts))[0]
#    return G
def L_func(t,N_pnts,M,xL,xR):
    L = np.zeros((M+1,M+1))
    for i in range(0,M+1):
        for j in range(0,M+1):
            if i > j and (i+j) % 2 !=0: 
                L[i,j] = integrate.quad(BjdBidx_func,xL,xR,args=(i,j,t,N_pnts,xL,xR))[0]
            else:
                L[i,j] = 0
    return L 
def M_func(t,N_pnts,M,xL,xR):
    MM = np.zeros((M+1,M+1))
    for i in range(0,M+1):
        for j in range(0,M+1):
            MM[i,j] = integrate.quad(BjBi_func,xL,xR,args=(i,j,t,N_pnts,xL,xR))[0]
    return MM
def LU_surf_func(u,space,mul,M,xL,xR):
    sumright = 0
    sumleft = 0
    for j in range(0,M+1):
        sumright += surf_func(mul,u,space,j,"R",xL,xR)
        sumleft += surf_func(mul,u,space,j,"L",xL,xR)
    LU = np.zeros(M+1).transpose()
    for i in range(0,M+1):
        LU[i] = Bi_func(xR,i,xL,xR)*((sumright)) - Bi_func(xL,i,xL,xR)*((sumleft))
#    if np.abs(LU[0])>1e4:
#        print("LU",LU)
#        print("Solution vector",V)
#        print(surf_func(mul,V,0,k,"L"))
    return LU 
def surf_func(mul,u,space,j,side,xL,xR):
#    print(side)
    B_right = Bi_func(xR,j,xL,xR)
    B_left = Bi_func(xL,j,xL,xR)
    if mul > 0 and side == "R":
        return u[space,j]*B_right
    elif mul > 0 and side =="L":
        if space !=0:
#            print(u[k-1,j])
            return u[space-1,j]*B_right 
        else:
            return 0
    elif mul < 0 and side =="R":
        if space != len(u[:,0])-1:
            return u[space+1,j]*B_left
        else:
            return 0
    elif mul < 0 and side =="L":
        return u[space,j]*B_left
    
#def GU_surf_func(U,k,mul,M,xL,xR,dxL,dxR):
#    GU_surf = np.zeros((1,M+1))
#    tempR = np.zeros((1,M+1))
#    tempL =  np.zeros((1,M+1))
#    for j in range(0,M+1):
#        tempR[j] = Bi_func(xR,j,xL,xR)*surf_func(mul,U,j,k,"R")
#        tempL[j] = Bi_func(xL,j,xL,xR)*surf_func(mul,U,j,k,"L")
#    for i in range(0,M+1):
#        right = dxR*Bi_func(i,xR,xL,xR)*np.sum(tempR,axis=1)
#        left = dxL*Bi_func(i,xL,xL,xR)*np.sum(tempL,axis=1)
#        GU_surf[i] = right-left
#    return GU_surf
def old_P_func(t,M,xL,xR):
     P = np.zeros((M+1,1))
     for i in range(0,M+1):
        result = integrate.quad(Bi_phi,xL,xR,args=(i,t,xL,xR))[0]
        P[i,0] = result
     return P
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
def phi_sol_func(t,u,N_space,N_ang,M,ws):
    div = 100                     #number of subdivisions in each zone 
    xs_list = np.zeros((N_space*div))
    phi_list = np.zeros((N_space*div))
    for k in range(0,N_space):
        result = grid_func(k,N_space,t)
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
        phi = phi_c + phi_u(xs,t)
#        phi_list[k*div:(k+1)*div] = phi
#        print(phi[0],phi[-1])
        phi_list[k*div+1:(k+1)*div-1] = phi[1:-1] 
        phi_list[k*div] += phi[0]
        phi_list[(k+1)*div-1] += phi[-1]
        
    return xs_list,phi_list 
def nodevals(t,N_ang,N_space,M,u,ws):
    psi_c = np.zeros((N_ang,N_space))
    center_list = np.zeros(N_space)
    for k in range(N_space):
        result = grid_func(k,N_space,t)
        xL = result[0]
        xR= result[1]
        center = result[4]
        center_list[k] = center
        for ang in range(N_ang):
            for j in range(M+1):
                psi_c[ang,k] += Bi_func(center,j,xL,xR)*u[ang,k,j]
    phi_c = np.sum(np.multiply(psi_c.transpose(),ws),axis=1) 
    return center_list,phi_c
def isotropic_DG_split_rhs(t,V,N_space,N_ang,mus,ws,LL,M,sigma_t,sigma_s,problem):
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
    for space in range(0,N_space):
        result = grid_func(space,N_space,t)
        xL = result[0]
        xR= result[1]
        phi_c = np.zeros(M+1).transpose()
        for i in range(0,M+1):
            phi_c[i]  = np.sum(np.multiply(V_old[:,space,i],ws))
#            print(np.multiply(V_old[:,space,j],ws))
#        phi_c = np.sum(np.multiply(V_new[:,space,0],ws),axis=0).transpose()
        for angle in range(N_ang):
            mul = mus[angle]
            L_surf = LU_surf_func(V_old[angle,:,:],space,mul,M,xL,xR)
            L = LL/(xR-xL)
            P = P_func(t,M,xL,xR)
            U = np.zeros(M+1).transpose()
            U[:] = V_old[angle,space,:]
            ###################################################################
#             testing new functions
#            L_test = L_func(t,N_space,M,xL,xR)
#            P_test = P_func(t,M,xL,xR)
##            if((P != P_test).all()):
##                print(P,P_test,xL,xR,t,"P functions")
#            if((L != L_test).all()):
#                print(L,L_test,xL,xR,t,"L functions")
#            if problem =="steady":
#                P = np.ones(M+1).transpose()
            RHS = mul*(-L_surf+ np.dot(L,U)) - sigma_t*U + sigma_s*phi_c + sigma_s*P
            V_new[angle,space,:] = RHS.transpose()
#            if problem =="steady":
#                 V_new[1:,0,:] = V_new[1:,-1,:] = 0
    return V_new.reshape(N_ang*N_space*(M+1))
def run_isotropic_DG(t=5,N_spaces=[1],N_angles=[256],Ms=[3],problem="ganapol"):
    sigma_s = 1
    sigma_t = 1
    for i in range(len(N_spaces)):
        N_space = N_spaces[i]
        N_ang = N_angles[i]
        M = Ms[i]
        mus = quadpy.c1.gauss_lobatto(N_ang).points
        ws = quadpy.c1.gauss_lobatto(N_ang).weights
        ws = ws/np.sum(ws)
        IC = np.zeros((N_ang,N_space,M+1)) 
        L = L_func(0,N_space,M,-1/2,1/2) 
        rhs = lambda t,V: isotropic_DG_split_rhs(t,V,N_space,N_ang,mus,ws,L,M,sigma_t,sigma_s,problem)
        sol = solve_ivp(rhs, [0.0,t], IC.reshape(N_ang*N_space*(M+1)), method='RK45')
        if not (sol.status == 0):
            print("solver failed %.0f"%N_space)
        sol_last = sol.y[:,-1].reshape((N_ang,N_space,M+1))
        xs,phi = phi_sol_func(t,sol_last,N_space,N_ang,M,ws) 
        center, nodes = nodevals(t,N_ang,N_space,M,sol_last,ws)
        plt.plot(xs,phi,"-")
        plt.scatter(center,nodes+phi_u(center,t),marker="x",label="M=%.0f N_space %.0f N_ang %.0f"%(M,N_space,N_ang))
        plt.legend()
#        plt.plot(center,np.sum(np.multiply(sol_last[:,:,0].transpose(),ws),axis=1),label="spaces%.0f"%N_space)
        plt.legend()
#        plt.ylim(-0.1,0.8)
#        print(sol_last)    
        if problem == "ganapol":
            if (t==1):
                sol = np.loadtxt("plane001.dat",delimiter="  ", usecols=[1,2])
                plt.plot(sol[:,0], sol[:,1],'k-')
                plt.plot(-sol[:,0], sol[:,1],'k-')
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
            plt.scatter(grid_func(k,N_space,t)[0],0,marker = "|",c="k")
            plt.scatter(grid_func(k,N_space,t)[1],0,marker = "|",c="k")
    return sol_last 
ganapol = run_isotropic_DG(t=1,N_spaces=[3],N_angles=[8],Ms=[3],problem="ganapol")