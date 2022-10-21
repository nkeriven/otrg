# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import torch

import utils.ot as uot
import utils.data as udata
import utils.plot as uplot

import pyvista as pv

###

plt.close('all')
np.random.seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

figpath = 'fig/'
stepsize_v = 4 # vertical discretization
stepsize_h = 4 # horizontal discretization

### data
n = 4000
X = udata.sphere(2*n, mode='hybrid', half=True)
n = X.shape[0] # after cutting half, n may have changed
dist = dict()
dist[0] = torch.tensor(udata.sphere_distrib(X, mode='cross', scale=.15), device=device)
dist[1] = torch.tensor(udata.sphere_distrib(X, mode='circle', scale=.15, c=np.array([1,1,1])) +
                       udata.sphere_distrib(X, mode='circle', scale=.15, c=np.array([-1,1,1])) +
                       udata.sphere_distrib(X, mode='circle', scale=.15, c=np.array([1,-1,1])) +
                       udata.sphere_distrib(X, mode='circle', scale=.15, c=np.array([-1,-1,1])),
                       device=device)
dist[1] /= dist[1].sum()
dist[2] = torch.tensor(udata.sphere_distrib(X, mode='outer_circle', scale=.95), device=device)
dist[3] = torch.tensor(udata.sphere_distrib(X, mode='circle', scale=.15), device=device)

### create plotting mesh (not random graph!)
cloud = pv.PolyData(X)
surf = cloud.delaunay_2d()
surf = surf.smooth()

### random graph and shortest path
G, h = udata.connected_eps_graph(X)

print('Compute shortest paths (can be long)...')
C, SP = uot.SP_matrix(G, device=device)

C = (h*C)**2 # here we use p=2

trueC = np.arccos(X @ X.T)
np.fill_diagonal(trueC, 0)
trueC = torch.tensor(trueC, device=device)**2

### plot function
def plot(bary, filename, cmap):
    uplot.pvplot(X,surf,bary,filename=filename,
                 focal_point=(0,0,0.2), zoom=1.5,
                 specular=.5, diffuse=2, ambient=.2)

### Wass barycenter
epsilon = .03
for (i,ti) in enumerate(np.linspace(0,1,stepsize_v)):
    for (j,tj) in enumerate(np.linspace(0,1,stepsize_h)):
        print(f'{i+1}/{stepsize_v}, {j+1}/{stepsize_h}')
        weights = np.array([ti*tj,
                            ti*(1-tj),
                            (1-ti)*tj,
                            (1-ti)*(1-tj)])
        weights /= weights.sum()

        if np.any(weights==1): # do not compute barycenter for pure distrib
            s = np.where(weights==1)[0].item()
            bary = dist[s]
        else:
            print('Sinkhorn...')
            bary, _ = uot.barycenters(C, dist, weights, device=device, epsilon=epsilon,
                                       n_iter=300, same_space=True)
        bary = bary.cpu().numpy()

        plot(X, bary, filename=figpath + f'barycenters_sphere{i}{j}.png',
             cmap=uplot.create_cmap(weights))

### Wass barycenter with true geodesic
for (i,ti) in enumerate(np.linspace(0,1,stepsize_v)):
    for (j,tj) in enumerate(np.linspace(0,1,stepsize_h)):
        print(f'{i+1}/{stepsize_v}, {j+1}/{stepsize_h}')
        weights = np.array([ti*tj,
                            ti*(1-tj),
                            (1-ti)*tj,
                            (1-ti)*(1-tj)])
        weights /= weights.sum()

        if np.any(weights==1):
            s = np.where(weights==1)[0].item()
            bary = dist[s]
        else:
            print('Sinkhorn...')
            bary, _ = uot.barycenters(trueC, dist, weights, device=device, epsilon=epsilon,
                                       n_iter=300, same_space=True)
        bary = bary.cpu().numpy()

        plot(X, bary, filename=figpath + f'barycenters_sphere{i}{j}true.png',
             cmap=uplot.create_cmap(weights))
