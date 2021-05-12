#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 10:17:08 2021

@author: ryanmcclarren
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 10:08:57 2021

@author: ryanmcclarren
"""
import numpy.polynomial.chebyshev as C
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.integrate import solve_ivp
from scipy.special import roots_legendre
from numba import jit, uintc, float64
import quadpy
from decimal import *
getcontext().prec = 28

@jit#( float64[:](uintc))
def cheby_gauss_lobatto(N,left=-1,right=1):
    """Define the zeros plus endpoints collocation points (Gauss-Lobatto quadrature points)
    for Chebyshev polynomials.
    
    N (integer): The number of points to generate
    left,right (floats): Left and right endpoints of the interval to perform 
                  collocation over. The default is -1, 1
    
    Returns
    The N collocation points cos(pi*n/(N-1)) n = 0..N-1. 
    The points are sorted from -1 to 1
    """
    ns = np.arange(0,N, dtype=np.float64)
    return np.sort(np.cos(np.pi/(N-1)*ns)*(right-left)*0.5 + (left+right)*0.5)
@jit
def cardinal_funcs(N,left=-1,right=1):
    """Define cardinal functions at the Chebyshev collocation points between a and b
    Cardinal functions are one at one collocation point and zero at the others
    
    N (integer): Number of cardinal functions/collocation points
    left,right (floats): Left and right endpoints of the interval to perform 
                  collocation over. The default is -1, 1
    
    Returns
    A list of the coefficients of the N cardinal polynomials.
    """
    a = []
    #get the collocation points
    xs = cheby_gauss_lobatto(N,left,right)
    for i in range(0,N+1):
        #do lagrange interpolation where the function is 1 at point i
        #and zero at the others.
        tmp = np.zeros((N+1),dtype=np.float64)
        tmp[i] = 1
        a.append(lagrange(xs,tmp))
    return a
@jit
def cardinal_derivs(N,a):
    """Compute the derivatives of the Cardinal functions at the collocation points.
    This function uses the polynomial derivative function of numpy.
    
    N (integer): number of points
    a (object): list containing the coefficients for each of the N Cardinal functions
    
    Returns
    aprime: A list of numpy polynomial objects for the derivatives of the Cardinal functions
    aprime_coef: A list of the polynomial coefficients for the derivtives
    """
    aprime = []
    aprime_coef = []
    for i in range(N+1):
        aprime.append(np.polyder(a[i]))
        aprime_coef.append(np.polyder(a[i]).coef)
    return aprime, aprime_coef

def colloc(x,N,nodeVals,a):
    """ Helper function for performing collocation at the points x using the 
    N cardinal functions contained in a with nodal values given by nodeVals
    
    x (numpy array): points to evaluate the function at
    N (integer): number of collocation points
    nodeVals (numpy array): value of the function at the N collocation points
    a (object): A list of numpy polynomial objects for the coefficients of the Cardinal functions
    
    Returns
    Value of the function at the points in x created from collocation    
    """
    tmp = np.multiply(np.array([a[i](x) for i in range(N) ]).transpose(),nodeVals)
    return np.sum(tmp,axis=1)

def d_colloc(x,N,nodeVals,a):
    """ Helper function for computing the derivative of the collocation at the points x using the 
    N cardinal functions contained in a with nodal values given by nodeVals
    
    x (numpy array): points to evaluate the function at
    N (integer): number of collocation points
    nodeVals (numpy array): value of the function at the N collocation points
    a (object): A list of numpy polynomial objects for the derivatives of the Cardinal functions
    
    Returns
    Value of the derivative of the function at the points in x created from collocation    
    """
    tmp = np.multiply(np.array([a[i](x) for i in range(N) ]).transpose(),nodeVals)
    return np.sum(tmp,axis=1)

def sweep_angle_pos(N,mu,q,bc,a,aprime):
    """
    Experimental: this was a test function to evaluate a transport sweep
    """
    A = np.identity(N-1)
    b = np.zeros(N-1)
    points = cheby_gauss_lobatto(N)
    #loop over the collocation points
    for point in range(1,N):
        #add mu times the derivative to the matrix
        b[point - 1] += -mu*aprime[0](points[point])*bc + q(points[point])
        for dpoint in range(1,N):
            A[point-1,dpoint-1] += mu*aprime[dpoint](points[point])
    sol = np.zeros(N)
    sol[0] = bc
    sol[1:] = np.linalg.solve(A,b)
    return sol


@jit
def ang_rhs(psi,N,mu,sigma,q,a,an,left=-1,right=1):
    """Function for computing the right-hand side of the transport equation at 
    the direction mu with a spatial collocation using Cardinal functions a,
    and their derivatives an.  Sigma and q are the total cross-section and source 
    nodal values in space.
    
    $$RHS = -\mu \partial_x \psi(x,\mu) - \sigma_\mathrm{t} \psi(x,\mu) + q$$
    
    psi (numpy array[N]): current value of the angular flux node values for direction mu
    N (integer): number of collocation points
    sigma (numpy array[N]): nodal values of the total cross-section
    q (numpy array[N]): nodal values of the source
    a (object): A list of numpy polynomial objects for the coefficients of the Cardinal functions
    an (object): A list of numpy polynomial objects for the derivatives of the Cardinal functions
    left,right (floats): Left and right endpoints of the interval to perform 
                  collocation over. The default is -1, 1
    Returns
    The N nodal values of the RHS of the transport equation
    """
    #RHS = -mu * d psi/dx -sigma psi + q 
    #compute RHS for angle mu
    #skip left point for mu>0 and right for mu<0 to not change BC
    eval_inds = np.arange(1,N,dtype=np.int)*(mu>0) + np.arange(0,N-1,dtype=np.int)*(mu<0)
    points = cheby_gauss_lobatto(N,left,right)[eval_inds]
    #evaluate part from q-sigma
    sol = colloc(points,N,q-sigma*psi,a)
    #evaluate mu*deriv part
    sol -= d_colloc(points,N,psi,an)*mu
    return sol

@jit
def isotropic_rhs(psi,N_space,N_ang,mus,ws,sigma,sigma_s,q,a,an,left=-1,right=1):
    """Function for computing the right-hand side of the transport equation 
    with isotropic scattering for every direction with a spatial collocation 
    using Cardinal functions a, and their derivatives an.  
    Sigma and q are the total cross-section and source 
    nodal values in space.
    
    $$RHS = -\mu_l \partial_x \psi(x,\mu_l) - \sigma_\mathrm{t} \psi(x,\mu_l) + \frac{\sigma_s}\sum_{i=0}{N_ang-1}w_i \psi(x,mu_i) +  q$$
    
    psi (numpy array[N_space*N_ang]): current value of the angular flux node values at all points in space and direction
    N_space (integer): number of spatial collocation points
    N_ang (integer): number of angular (direction) collocation points
    mus (numpy array[N_ang]): The collocation in direction points
    ws (numpy array[N_ang]): The weights for the quadrature rule defined by mus (assumed to be normalized to sum to 1)
    sigma (numpy array[N]): nodal values of the total cross-section
    sigma_s (numpy array[N]): nodal values of the scattering cross-section
    q (numpy array[N]): nodal values of the source
    a (object): A list of numpy polynomial objects for the coefficients of the Cardinal functions
    an (object): A list of numpy polynomial objects for the derivatives of the Cardinal functions
    left,right (floats): Left and right endpoints of the interval to perform 
                  collocation over. The default is -1, 1
    Returns
    The NM nodal values of the RHS of the transport equation for all collocation and mu points
    """
    #psi is laid out as N_ang by N_space
    #compute scalar flux at each point
    psi_new = psi.copy().reshape((N_ang,N_space))
    phi = np.sum(np.multiply(psi_new.transpose(),ws),axis=1)
    source = sigma_s*phi + q
    for angle in range(N_ang):
        if (mus[angle] > 0):
            psi_new[angle,1:] = ang_rhs(psi_new[angle,:], N_space, mus[angle],sigma,source,a,an, left=left,right=right)
            psi_new[angle,0] = 0
        else:
            psi_new[angle,0:N_space-1] = ang_rhs(psi_new[angle,:], N_space, mus[angle],sigma,source,a,an, left=left,right=right)
            psi_new[angle,-1] = 0
    return psi_new.reshape(N_ang*N_space)


@jit
def p1_rhs(psi,N_space,N_ang,mus,ws,sigma,sigma_s,sigma_s1,q,a,an,left=-1,right=1):
    """Function for computing the right-hand side of the transport equation 
    with linear anisotropic scattering for every direction with a spatial collocation 
    using Cardinal functions a, and their derivatives an.  
    Sigma and q are the total cross-section and source 
    nodal values in space.
    
    $$RHS = -\mu_l \partial_x \psi(x,\mu_l) - \sigma_\mathrm{t} \psi(x,\mu_l) + \frac{\sigma_s}\sum_{i=0}{N_ang-1}w_i \psi(x,mu_i) + \frac{\sigma_{s1}\mu_l}\sum_{i=0}{N_ang-1}w_i\mu_i \psi(x,mu_i) + q$$
    
    psi (numpy array[N_space*N_ang]): current value of the angular flux node values at all points in space and direction
    N_space (integer): number of spatial collocation points
    N_ang (integer): number of angular (direction) collocation points
    mus (numpy array[N_ang]): The collocation in direction points
    ws (numpy array[N_ang]): The weights for the quadrature rule defined by mus
    sigma (numpy array[N]): nodal values of the total cross-section
    sigma_s (numpy array[N]): nodal values of the zeroth moment of the differential scattering cross-section
    sigma_s1 (numpy array[N]): nodal values of the first moment of the differential scattering cross-section
    q (numpy array[N]): nodal values of the source
    a (object): A list of numpy polynomial objects for the coefficients of the Cardinal functions
    an (object): A list of numpy polynomial objects for the derivatives of the Cardinal functions
    left,right (floats): Left and right endpoints of the interval to perform 
                  collocation over. The default is -1, 1
    Returns
    The NM nodal values of the RHS of the transport equation for all collocation and mu points
    """
    #psi is laid out as N_ang by N_space
    #compute scalar flux at each point
    psi_new = psi.copy().reshape((N_ang,N_space))
    phi = np.sum(np.multiply(psi_new.transpose(),ws),axis=1)
    J = np.sum(np.multiply(psi_new.transpose(),ws*mus),axis=1)
    source = 0.5*sigma_s*phi + q
    source_1 = 0.5*sigma_s1*J
    for angle in range(N_ang):
        if (mus[angle] > 0):
            psi_new[angle,1:] = ang_rhs(psi_new[angle,:], N_space, mus[angle],sigma,source + mus[angle]*source_1,a,an, left=left,right=right)
            psi_new[angle,0] = 0
        else:
            psi_new[angle,0:N_space-1] = ang_rhs(psi_new[angle,:], N_space, mus[angle],sigma,source+ mus[angle]*source_1,a,an, left=left,right=right)
            psi_new[angle,-1] = 0
    return psi_new.reshape(N_ang*N_space)

@jit
def wynn_epsilon(data):
    """
    Experimental! Attempt to compute apply Wynn's epsilon method to accelerate convergence
    this requires high precision calculations to apply well.
    """
    output = []
    output.append(data)
    N = len(data)
    cols = N
    for column in range(1,cols):
        column_data = []
        for row in range(0,cols-column):
            if (column > 1):
                column_data.append(Decimal(output[column-2][row+1]) + Decimal(1)/(Decimal(output[column-1][row+1])- Decimal(output[column-1][row])))
            else:
                column_data.append(Decimal(1)/(Decimal(output[column-1][row+1])- Decimal(output[column-1][row])))
        output.append(column_data)
    return output


def run_GarciaSiewert(s_input=1,N_spaces = [4, 8, 12, 14, 16, 18,  20],    N_angles = [256, 256,  256,  256,   256,  256,  256]):
    """
    Return the problem from 
    Garcia,Siewert (1982). Radiative transfer in finite inhomogeneous plane-parallel atmospheres. Journal of Quantitative Spectroscopy and Radiative Transfer, 27(2), 141–148. http://doi.org/10.1016/0022-4073(82)90134-0
    """
    data = []
    data_r = []
    s = s_input
    for i in range(len(N_spaces)):
        N_space = N_spaces[i]
        N_ang = N_angles[i]
        sigma = np.ones(N_space)
        if (s<=1000):
            sigma_s = np.exp((-cheby_gauss_lobatto(N_space, left=0, right=5))/s) 
            s_print = s 
        else:
            s_print = "infinity"
            sigma_s = np.ones(N_space)
        q = np.zeros(N_space,dtype=np.float64)
        a = cardinal_funcs(N_space, left=0, right=5)
        an, ad = cardinal_derivs(N_space,a)
        mus = quadpy.c1.gauss_lobatto(N_ang).points
        ws = quadpy.c1.gauss_lobatto(N_ang).weights
        ws = ws/np.sum(ws)
        rhs = lambda t,psi: isotropic_rhs(psi,N_space,N_ang,mus,ws,sigma,sigma_s,q,a,an, left=0, right=5)
        IC = np.zeros((N_ang,N_space),dtype=np.float64)
        #incoming on right 
        IC[mus>0, 0] = 1.0
        sol = solve_ivp(rhs,[0,1000.0],IC.reshape(N_ang*N_space),method="BDF")
        sol_last = sol.y[:,-1].reshape((N_ang,N_space))
        np.save("GS_" + str(s_print) + "_Nx_" + str(N_space) + "_Nang_" + str(N_ang) + "_psi.npz", sol_last)
        data.append(sol_last[0,0])
        data_r.append(sol_last[-1,-1])
        mus_bench = np.array([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        if (s==1):
            plot = 1
            s1_left_bench = [0.58966,0.53112,0.44328,0.38031,0.33296,0.29609,0.26656,0.24239,0.22223,0.20517,0.19055]
            s1_right_bench = [6.0750E-06,6.9252E-06,9.6423E-06,1.6234E-05,4.3858E-05,1.6937E-04,5.7347E-04,1.5128E-03,3.2437E-03,5.9605E-03,9.7712E-03]
            plt.plot(-mus_bench,s1_left_bench,'o')
            plt.plot(mus[mus<0],sol_last[mus<0,0])
            plt.show()
            plt.plot(mus_bench,s1_right_bench,'o')
            plt.semilogy(mus[mus>0],sol_last[mus>0,-1])
            plt.show()
        elif (s_print == "infinty"):
            plot = 1
            sinf_right_bench = [0.102200,0.112160,0.130420,0.147700,0.164500,0.181000,0.197320,0.213510,0.229570,0.245500,0.261200]
            sinf_left_bench = [0.8978,0.88784,0.86958,0.8523,0.8355,0.819,0.80268,0.78649,0.77043,0.7545,0.73872]
            plt.plot(-mus_bench,sinf_left_bench,'o')
            plt.plot(mus[mus<0],sol_last[mus<0,0])
            plt.show()
            plt.plot(mus_bench,sinf_right_bench,'o')
            plt.semilogy(mus[mus>0],sol_last[mus>0,-1])
            plt.show()
        print(wynn_epsilon(data))
        print("Right\n===================")
        print(wynn_epsilon(data_r))
    return data,data_r, sol_last
def run_ganapol(t=1,left=-1,right=1,sd = 0.1,N_angles = [256, 256,  256,  256,   256,  256,  256], N_spaces = [4, 8, 12, 14, 16, 18,  20]):
    """
    Run an approximation to Ganapol's plane source problem. We use a Gaussian with standard deviation sd rather than a delta function.
    
    t (float): final time
    left,right (floats): Left and right endpoints of the domain. The default is -1, 1
    sd (float): standard deviation of the Gaussian initial condition    
    """
    data = []
    data_r = []
    for i in range(len(N_spaces)):
        N_space = N_spaces[i]
        N_ang = N_angles[i]
        sigma = np.ones(N_space)
        sigma_s = sigma.copy()
        q = np.zeros(N_space,dtype=np.float64)
        a = cardinal_funcs(N_space,left,right)
        an, ad = cardinal_derivs(N_space,a)
        mus = quadpy.c1.gauss_lobatto(N_ang).points
        ws = quadpy.c1.gauss_lobatto(N_ang).weights
        ws = ws/np.sum(ws)
        rhs = lambda t,psi: isotropic_rhs(psi,N_space,N_ang,mus,ws,sigma,sigma_s,q,a,an,left,right)
        IC = np.zeros((N_ang,N_space),dtype=np.float64)
        for i in range(N_ang):
            IC[i,:] = np.exp(-cheby_gauss_lobatto(N_space,left,right)**2/2/sd**2)/(sd*np.sqrt(2*np.pi))
        IC[mus>0,0] = 0
        IC[mus<0,-1] = 0
        sol = solve_ivp(rhs,[0,t],IC.reshape(N_ang*N_space),method="DOP853")
        sol_last = sol.y[:,-1].reshape((N_ang,N_space))
        np.save("Ganapol_bounds" + str(left) + "_" + str(right) + "_Nx_" + str(N_space) + "_Nang_" + str(N_ang) + "_psi.npz", sol_last)
        data.append(sol_last[0,0])
        data_r.append(sol_last[-1,-1])
        
        xs = np.linspace(left,right,200)
        phi = np.sum(np.multiply(sol_last.transpose(),ws),axis=1)
        plt.plot(xs,colloc(xs,N_space,phi,a))
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
        plt.show()
    return data,data_r, sol_last
def run_lin_aniso(t=100,c1=0,t0=1,N_angles = [256, 256,  256,  256,   256,  256,  256], N_spaces = [4, 8, 12, 14, 16, 18,  20]):
    """
    Run an anisotropic scattering problem from 
    M. Razzaghi, S. Oppenheimer & F. Ahmad (2002) A Legendre Wavelet Method for the Radiative Transfer Equation in Remote Sensing, Journal of Electromagnetic Waves and Applications, 16:12, 1681-1693, DOI: 10.1163/156939302X00507
    
    t (float): final time (this is a steady state problem so should be large)
    c1 (float): value of sigma_s1 in the problem   
    t0 (float): thickness of the slab (called t0 in the paper)
    
    Returns
    data: list of values of 2 times the exiting current, J+
    sol: the last angular flux solution computed
    """
    #N_spaces = [4, 8, 12, 14, 16, 18,  20]
    #N_angles = [256, 256,  256,  256,   256,  256,  256]
    data = []
    data_r = []
    for i in range(len(N_spaces)):
        N_space = N_spaces[i]
        N_ang = N_angles[i]
        sigma = np.ones(N_space)
        sigma_s = sigma.copy()
        sigma_s1 = sigma.copy()*c1
        q = np.zeros(N_space,dtype=np.float64)
        a = cardinal_funcs(N_space, left=0, right = t0)
        an, ad = cardinal_derivs(N_space,a)
        mus = quadpy.c1.gauss_lobatto(N_ang).points
        ws = quadpy.c1.gauss_lobatto(N_ang).weights
        #ws = ws/np.sum(ws)
        rhs = lambda t,psi: p1_rhs(psi,N_space,N_ang,mus,ws,sigma,sigma_s,sigma_s1,q,a,an, left=0, right = t0)
        IC = np.zeros((N_ang,N_space),dtype=np.float64)
        IC[mus>0,0] = 1
        IC[mus<0,-1] = 0
        sol = solve_ivp(rhs,[0,t],IC.reshape(N_ang*N_space),method="BDF")
        sol_last = sol.y[:,-1].reshape((N_ang,N_space))
        np.save("linAniso_" + str(c1) + "_Nx_" + str(N_space) + "_Nang_" + str(N_ang) + "_psi.npy", sol_last)
        data.append(np.sum(np.multiply(sol_last[mus>0,-1].transpose(),ws[mus>0]*mus[mus>0]))*2)
        
        print(wynn_epsilon(data))
    return data, sol_last
def run_quadratic(t=1,N_angles = [256, 256,  256,  256,   256,  256,  256], N_spaces = [4, 8, 12, 14, 16, 18,  20]):
    """
    Similar to Ganapols problem but the initial condition is a quadratic in the middle
    
    t (float): final time
    """
    data = []
    data_r = []
    for i in range(len(N_spaces)):
        N_space = N_spaces[i]
        N_ang = N_angles[i]
        sigma = np.ones(N_space)
        sigma_s = sigma.copy()
        q = np.zeros(N_space,dtype=np.float64)
        a = cardinal_funcs(N_space)
        an, ad = cardinal_derivs(N_space,a)
        mus = quadpy.c1.gauss_lobatto(N_ang).points
        ws = quadpy.c1.gauss_lobatto(N_ang).weights
        ws = ws/np.sum(ws)
        rhs = lambda t,psi: isotropic_rhs(psi,N_space,N_ang,mus,ws,sigma,sigma_s,q,a,an)
        IC = np.zeros((N_ang,N_space),dtype=np.float64)
        for i in range(N_ang):
            IC[i,:] = 3-2*(cheby_gauss_lobatto(N_space))**2
        IC[mus>0,0] = 0
        IC[mus<0,-1] = 0
        sol = solve_ivp(rhs,[0,t],IC.reshape(N_ang*N_space),method="DOP853",rtol=1e-5, atol=1e-8)
        sol_last = sol.y[:,-1].reshape((N_ang,N_space))
        np.save("quadratic"  + "_Nx_" + str(N_space) + "_Nang_" + str(N_ang) + "_psi.npz", sol_last)
        
        phi = np.sum(np.multiply(sol_last.transpose(),ws),axis=1)
        data.append(phi[0])
        data_r.append(colloc([0],N_space,phi,a)[0])
        out = np.vstack((xs,colloc(xs,N_space,phi,a))).transpose()
        np.save("quadratic" + "_Nx_" + str(N_space) + "_Nang_" + str(N_ang) + "_phi.npy", out)
        plt.plot(xs,colloc(xs,N_space,phi,a))
        plt.show()
        print(wynn_epsilon(data))
        print("Right\n===================")
        print(wynn_epsilon(data_r))
    return data,data_r, sol_last
if __name__ == "__main__":
    run_GarciaSiewert(N_angles=[16], N_spaces=[6], s=1)
    