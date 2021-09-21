import numpy as np
import matplotlib.pyplot as plt
import h5py


# =============================================================================
# Plot results
# =============================================================================

# Load grids
with h5py.File('output.h5', 'r') as f:
    x = f['tally/spatial_grid'][:]
    t = f['tally/time_grid'][:]
dx = x[1:]-x[:-1]
K = len(t)-1

dx    = (x[1]-x[0])
x_mid = 0.5*(x[:-1]+x[1:])
with h5py.File('output.h5', 'r') as f:
    phi_edge    = f['tally/flux-edge/mean'][:]/dx
    phi_edge_sd = f['tally/flux-edge/sdev'][:]/dx

# Plot
for k in range(K):
    plt.figure(6)
    plt.plot(x_mid,phi_edge[k],'--k',label="MC")
    plt.fill_between(x_mid,phi_edge[k]-phi_edge_sd[k],phi_edge[k]+phi_edge_sd[k],alpha=0.2,color='b')
    plt.xlabel(r'$x$, cm')
    plt.ylabel('Flux')
    plt.xlim([max(x[0],-t[k+1]*1.6),min(t[k+1]*1.6,x[-1])])
    plt.grid()
    plt.legend()
    plt.title(r'$\bar{\phi}_i(t=%.1f)$'%t[k+1])
    plt.show()
