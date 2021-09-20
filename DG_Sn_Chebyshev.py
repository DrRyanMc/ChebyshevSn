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
from numba import jit, uintc, float64, njit, prange 
import quadpy
#from decimal import *
from scipy.special import eval_legendre, lpmv
import math
from scipy.interpolate import interp1d
# @jit(float64(float64,float64,float64))
def Bi_func(x,i,xL,xR):
    # orthonormal basis functions 
    # return (math.sqrt(1 + 2*i)*eval_legendre(i,(-2*x + xL + xR)/(xL - xR)))/math.sqrt(-xL + xR)
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
    return Bi_func(x,i,xL,xR)*np.exp(-t)*np.heaviside((t+x)/t,0)*np.heaviside(1-x/t,0)*np.heaviside(1+x,0)*np.heaviside(1-x,0)/(2*t)
def phi_u(x,tt,dx):
    t = tt 
    return np.exp(-t)*np.heaviside((t+x)/t,0)*np.heaviside(1-x/t,0)/(2*t+dx)
def phi_u_finite(x,tt,dx):
    t = tt
    return np.exp(-t)*np.heaviside((t+x)/t,0)*np.heaviside(1-x/t,0)*np.heaviside(1+x,0)*np.heaviside(1-x,0)/(2*t+dx)
@jit
def grid_func(k,N_space,t,left,right,dx,mus,tfinal,mode):
    if mode == "static" or mode == "finite2":
        Leftedge = -tfinal
        Rightedge = tfinal
        D = Rightedge-Leftedge
        dxk = D/N_space
        xL = Leftedge + k*dxk
        xR = Leftedge+(k+1)*dxk
        center = (xL+xR)/2
        # print(xL,xR)
        ############################
        dxL = 0
        dxR = 0
        ############################
    elif mode == "sqrt":
        speed = np.max(mus)
        if (k < N_space//2):
            xL = left[k] + -1*t*(speed)*math.sqrt(left[k]/left[0]) #bt[i](i,t)
            dxL = -1*speed*math.sqrt(left[k]/left[0])
            xR = right[k] + -1*t*(speed)*math.sqrt(right[k]/left[0])
            dxR = -1*speed*math.sqrt(right[k]/left[0])
        elif (k>= N_space//2):
            xL = left[k] + 1*t*np.max(mus)*math.sqrt(left[k]/right[N_space-1])
            dxL = 1*np.max(mus)*math.sqrt(left[k]/right[N_space-1])
            xR = right[k] + 1*t*np.max(mus)*math.sqrt(right[k]/right[N_space-1])
            dxR = 1*np.max(mus)*math.sqrt(right[k]/right[N_space-1])
    elif mode == "linear":
        speed = np.max(mus)
        if (k < N_space//2):
            xL = left[k] + -1*t*(speed)*left[k]/left[0] #bt[i](i,t)
            dxL = -1*speed*left[k]/left[0]
            xR = right[k] + -1*t*(speed)*right[k]/left[0]
            dxR = -1*speed*right[k]/left[0]
        elif (k>= N_space//2):
            xL = left[k] + 1*t*np.max(mus)*left[k]/right[N_space-1]
            dxL = 1*np.max(mus)*left[k]/right[N_space-1]
            xR = right[k] + 1*t*np.max(mus)*right[k]/right[N_space-1]
            dxR = 1*np.max(mus)*right[k]/right[N_space-1]
    elif mode == "finite":
        speed = np.max(mus)
        if (t <= 1):
            if (k < N_space//2):
                xL = left[k] + -1*t*(speed)*left[k]/left[0] #bt[i](i,t)
                dxL = -1*speed*left[k]/left[0]
                xR = right[k] + -1*t*(speed)*right[k]/left[0]
                dxR = -1*speed*right[k]/left[0]
            elif (k>= N_space//2):
                xL = left[k] + 1*t*np.max(mus)*left[k]/right[N_space-1]
                dxL = 1*np.max(mus)*left[k]/right[N_space-1]
                xR = right[k] + 1*t*np.max(mus)*right[k]/right[N_space-1]
                dxR = 1*np.max(mus)*right[k]/right[N_space-1]
        else:
            if (k < N_space//2):
                xL = left[k] + -1*1*(speed)*left[k]/left[0] #bt[i](i,t)
                dxL = 0
                xR = right[k] + -1*1*(speed)*right[k]/left[0]
                dxR = 0
            elif (k>= N_space//2):
                xL = left[k] + 1*1*np.max(mus)*left[k]/right[N_space-1]
                dxL = 0
                xR = right[k] + 1*1*np.max(mus)*right[k]/right[N_space-1]
                dxR = 0    
    center = (xL+xR)/2
    return xL,xR,dxL,dxR,center
@jit
def G_func(k,t,N_space,M,dx,xL,xR,dxL,dxR):
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
def LU_surf_func(u,space,N_space,mul,M,xL,xR,dxL,dxR,mode):
    sumright = 0
    sumleft = 0
    rightspeed = mul - dxR
    leftspeed = mul-dxL
    for j in range(0,M+1):
        sumright += surf_func(rightspeed,u,space,j,"R",xL,xR,N_space)
        sumleft += surf_func(leftspeed,u,space,j,"L",xL,xR,N_space)
    # else: 
    #     for j in range(0,M+1):
    #         sumright += surf_func_finite2(rightspeed,u,space,j,"R",xL,xR,N_space)
    #         sumleft += surf_func_finite2(leftspeed,u,space,j,"L",xL,xR,N_space)
        
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
    center = (xL+xR)/2
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
@jit 
def surf_func_finite2(speed,u,space,j,side,xL,xR,N_space,):
    center = (xL+xR)/2
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
    if -1 <= center <= 1:
        if speed > 0 and side == "R" :
            return u[space,j]*B_right
        elif speed > 0 and side =="L" and (space !=0):
            return u[space-1,j]*B_right 
        elif speed < 0 and side =="R" and (space != N_space-1):
            return u[space+1,j]*B_left
        elif speed < 0 and side =="L":
            return u[space,j]*B_left
        else:
            return 0
# @jit 
def P_func(t,M,xL,xR,mode):
    P = np.zeros(M+1).transpose()
    if mode == "static" or mode =="finite2":
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
def phi_sol_func(t,u,N_space,N_ang,M,ws,mus,dx,left,right,mode):
    div = (M+1)                   #number of subdivisions in each zone 
    xs_list = np.zeros((N_space*div))
    phi_list = np.zeros((N_space*div))
    j_list = np.zeros((N_space*div))
    for k in range(0,N_space):
        result = grid_func(k,N_space,t,left,right,dx,mus,t,mode)
        xL = result[0]
        xR= result[1]
        scheme = quadpy.c1.gauss_lobatto(M+1)
        xs = scheme.points
        xs_list[k*div:(k+1)*div] = xL + (xs+1)/(2)*(xR-xL)
        sol_vec = np.zeros((N_ang,M+1,div))
        for ang in range(N_ang):
            for j in range(0,M+1):
                for ix in range(0,div):
                    sol_vec[ang,j,ix] = math.sqrt(2*j+1)*lpmv(0,j,xs[ix])*u[ang,k,j]/math.sqrt(xR-xL)
        psi_c = np.zeros((N_ang,div)) 
        for ang2 in range(0,N_ang):
            psi_c[ang2,:] = np.sum(sol_vec[ang2,:,:],axis=0)
        phi_c = np.sum(np.multiply(psi_c.transpose(),ws),axis=1) 
        if mode != "finite2" and mode != "finite":
            phi = phi_c + np.exp(-t)/(2*t + dx)
        else:
            x = xs_list[k*div:(k+1)*div]
            phi = phi_c + phi_u_finite(x, t, dx)
        # jj = np.sum(np.multiply(psi_c.transpose(),ws*mus),axis=1)/phi_c
#        phi_list[k*div:(k+1)*div] = phi
#        print(phi[0],phi[-1])
        phi_list[k*div:(k+1)*div] = phi 
        # phi_list[k*div] += phi[0]
        # phi_list[(k+1)*div-1] += phi[-1]
        # j_list[k*div+1:(k+1)*div-1] = jj[1:-1] 
        # j_list[k*div] += jj[0]
        # j_list[(k+1)*div-1] += jj[-1] 

    return xs_list,phi_list,j_list  

def pointsol_func(u,x,t,N_ang,N_space,ws,M,left,right,dx,mus,tfinal,mode):
    # find the space the point is in 
    for k in range(N_space):
        result = grid_func(k,N_space,t,left,right,dx,mus,tfinal,mode)
        xL = result[0]
        xR= result[1]
        if (xL <= x) and (xR >= x):
            space = k 
    result = grid_func(space,N_space,t,left,right,dx,mus,tfinal,mode)
    xL = result[0]
    xR= result[1]
    psi_c = np.zeros(N_ang)
    for ang in range(N_ang):
        for j in range(M+1):
            psi_c[ang] += Bi_func(x,j,xL,xR)*u[ang,space,j]
    phi_c = np.sum(np.multiply(psi_c.transpose(),ws),axis=0)     
    phi = phi_c + phi_u(x,t,dx)
    # plt.scatter(x,phi)
    return phi 
@jit
def isotropic_DG_split_rhs(t,V,N_space,N_ang,mus,ws,LL,M,sigma_t_list,sigma_s_list,dx,left,right,problem,mode,tfinal):
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
    for space in prange(0,N_space):
        sigma_t = sigma_t_list[space]
        sigma_s = sigma_s_list[space]
        result = grid_func(space,N_space,t,left,right,dx,mus,tfinal,mode)
        xR= result[1]
        xL = result[0]
        dxL = result[2]
        dxR = result[3]
        phi_c = np.zeros(M+1).transpose()
        L = LL/(xR-xL)
        G = G_func(space,t,N_space,M,dx,xL,xR,dxL,dxR)
        P = np.zeros(M+1).transpose()
        if mode == "linear" or mode =="sqrt" or mode == "finite":
            P[0] = (math.exp(-t+dx)/(2*t+dx)*math.sqrt(xR-xL))
        for i in range(0,M+1):
            phi_c[i]  = np.sum(np.multiply(V_old[:,space,i],ws))
        for angle in range(N_ang):
            mul = mus[angle]
            L_surf = LU_surf_func(V_old[angle,:,:],space,N_space,mul,M,xL,xR,dxL,dxR,mode)
            U = np.zeros(M+1).transpose()
            U[:] = V_old[angle,space,:]
            ###################################################################
            RHS = np.dot(G,U) + -L_surf+ mul*np.dot(L,U) - sigma_t*U + sigma_s*phi_c + sigma_s*P
            V_new[angle,space,:] = RHS.transpose()
    return V_new.reshape(N_ang*N_space*(M+1))
def isotropic_DG_split_staticgrid_rhs_(t,V,N_space,N_ang,mus,ws,LL,M,sigma_t_list,sigma_s_list,dx,left,right,problem,mode,tfinal):
    # since numba doesn't like scipy, I made a new RHS for when the grid is not moving 
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
    for space in prange(0,N_space):
        sigma_t = sigma_t_list[space]
        sigma_s = sigma_s_list[space]
        result = grid_func(space,N_space,t,left,right,dx,mus,tfinal,mode)
        xR= result[1]
        xL = result[0]
        dxL = result[2]
        dxR = result[3]
        phi_c = np.zeros(M+1).transpose()
        L = LL/(xR-xL)
        P = P_func(t, M, xL, xR,mode)
        for i in range(0,M+1):
            phi_c[i]  = np.sum(np.multiply(V_old[:,space,i],ws))
        for angle in range(N_ang):
            mul = mus[angle]
            L_surf = LU_surf_func(V_old[angle,:,:],space,N_space,mul,M,xL,xR,dxL,dxR,mode)
            U = np.zeros(M+1).transpose()
            U[:] = V_old[angle,space,:]
            ###################################################################
            RHS = -L_surf+ mul*np.dot(L,U) - sigma_t*U + sigma_s*phi_c + sigma_s*P
            V_new[angle,space,:] = RHS.transpose()
    return V_new.reshape(N_ang*N_space*(M+1))
def run_isotropic_DG(tfinal=1,N_spaces=[2], M = 3, problem="ganapol", mode ="linear", weights = "gauss_lobatto"):
    plt.figure(6)
    if problem == "ganapol":
            if (tfinal==1):
                pl = np.loadtxt("plane001.dat",delimiter="  ", usecols=[1,2])
                plt.plot(pl[:,0], pl[:,1],'k--')
                plt.plot(-pl[:,0], pl[:,1],'k--')
            elif (tfinal==5):
                pl = np.loadtxt("plane005.dat",delimiter="  ", usecols=[1,2])
                plt.plot(pl[:,0], pl[:,1],'k-')
                plt.plot(-pl[:,0], pl[:,1],'k-')
            elif (tfinal==10):
                pl = np.loadtxt("plane010.dat",delimiter="  ", usecols=[1,2])
                plt.plot(pl[:,0], pl[:,1],'k-')
                plt.plot(-pl[:,0], pl[:,1],'k-')
    N_angles = []
    testpoints = [0.01,0.1,0.25,0.5,0.75,0.9]
    testmatrix = np.zeros((len(N_spaces) + 2, len(testpoints),))
    RMSdata = np.zeros((2,len(N_spaces)))
    testmatrix[0] = testpoints
    RMSdata[0] = N_spaces
    if tfinal == 1 or tfinal == 5 or tfinal == 10: 
        for it in range(len(testpoints)):
            index = np.argmin(np.abs(pl[:,0] - testpoints[it]))
            testmatrix[1,it] = pl[index,1]
    for ang in range(len(N_spaces)):
        N_angles.append(int(N_spaces[ang]*(2**(M+1))))
    errRMS = np.zeros(len(N_spaces))
    if mode == "linear" or mode =="finite":
        dx = 1e-16
    elif mode == "sqrt":
        dx = 1e-12
    else:
        dx = 0 
    for i in range(len(N_spaces)):
        N_space = N_spaces[i]
        N_ang = N_angles[i]
        if weights == "gauss_lobatto":
            mus = quadpy.c1.gauss_lobatto(N_ang).points
            ws = quadpy.c1.gauss_lobatto(N_ang).weights
        elif weights == "newton_cotes":
            mus = quadpy.c1.newton_cotes_closed(N_ang-1).points
            ws = quadpy.c1.newton_cotes_closed(N_ang-1).weights
        ws = ws/np.sum(ws)
        print(mus)
        sigma_s_list = np.ones(N_space)
        sigma_t_list = np.ones(N_space)
        hx = dx/N_space
        left = np.linspace(-dx/2, dx/2-hx, N_space)
        right = np.linspace(-dx/2+hx, dx/2, N_space)
        IC = np.zeros((N_ang,N_space,M+1)) 
        L = L_func(0,N_space,M,-1/2,1/2) 
        if mode == "finite2":
            for k in range(N_space):
                result = grid_func(k,N_space,tfinal,left,right,dx,mus,tfinal,mode)
                xR= result[1]
                xL = result[0]
                if xL >= 1 or xR <=-1:
                     sigma_t_list[k] = 0
                     sigma_s_list[k] = 0 
        if mode == "static" or mode == "finite2":
            rhs = lambda t,V: isotropic_DG_split_staticgrid_rhs_(t, V, N_space, N_ang, mus, ws, L, M, sigma_t_list, sigma_s_list, dx, left, right, problem, mode, tfinal)
        elif mode == "linear" or mode == "sqrt" or mode == "finite":
            rhs = lambda t,V: isotropic_DG_split_rhs(t, V, N_space, N_ang, mus, ws, L, M, sigma_t_list, sigma_s_list, dx, left, right, problem, mode, tfinal)
        sol = integrate.solve_ivp(rhs, [0.0,tfinal], IC.reshape(N_ang*N_space*(M+1)), method='RK45', t_eval = [tfinal], rtol = 10e-8, atol = 1e-6)
        if not (sol.status == 0):
            print("solver failed %.0f"%N_space)
        sol_last = sol.y[:,-1].reshape((N_ang,N_space,M+1))
        xs,phi,j = phi_sol_func(tfinal,sol_last,N_space,N_ang,M,ws,mus,dx,left,right,mode) 
        save_data_phi = np.zeros((2,len(phi)))
        save_data_phi[0] = xs
        save_data_phi[1] = phi
        if tfinal == 1 or tfinal == 5 or tfinal == 10: 
            sol_ganapol = interp1d(pl[:,0],pl[:,1], kind="cubic") 
            errRMS[i] = np.sqrt(np.mean((phi  - sol_ganapol(np.abs(xs)))**2))
        RMSdata[1] = errRMS
        if mode =="static":
            np.save("ganapol_t=%.0f_%.0f_spaces_%.0f_angles_M_%.0f_staticgrid"%(tfinal,N_space,N_ang,M), save_data_phi)
            np.save("errRMS_stat_M=%.0f_tfinal=%.0f"%(M,tfinal), RMSdata)
            np.save("testpoints_stat_M=%.0f_tfinal=%.0f"%(M,tfinal),testmatrix)
        elif mode == "linear":
            np.save("ganapol_t=%.0f_%.0f_spaces_%.0f_angles_M_%.0f_lineargrid"%(tfinal,N_space,N_ang,M), save_data_phi)
            np.save("errRMS_lin_M=%.0f_tfinal=%.0f"%(M,tfinal), RMSdata)
            np.save("testpoints_lin_M=%.0f_tfinal=%.0f"%(M,tfinal),testmatrix)
        for jt in range(len(testpoints)):
            x = testpoints[jt]
            phipoint = pointsol_func(sol_last, x, tfinal, N_ang, N_space, ws, M, left, right, dx, mus, tfinal, mode)  
            testmatrix[i+2,jt] = phipoint
        plt.plot(xs,phi,"-", label = "time %.1f"%tfinal)
        plt.legend()
        for k in range(0,N_space):
            plt.scatter(grid_func(k,N_space,tfinal,left,right,dx,mus,tfinal,mode)[0],0,marker = "|",c="k")
            plt.scatter(grid_func(k,N_space,tfinal,left,right,dx,mus,tfinal,mode)[1],0,marker = "|",c="k"),
        plt.show()
    print("Spaces = ", N_spaces,"ERROR RMS = ", errRMS)
    return sol_last
run_isotropic_DG(tfinal = 1, N_spaces = [2], M = 1, problem ="ganapol", mode = "linear", weights = "newton_cotes")