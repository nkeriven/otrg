# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 10:21:22 2021

@author: nicol
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import utils_ot as uot
import torch

import gif

plt.close('all')
np.random.seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

fontsize = 22
figsize = (6,4)
rc_fonts = {
    "text.usetex": True,
    "font.size": 14
}
plt.rcParams.update(rc_fonts)

size = 10 # discretization of the weights
n = 1000
epsilon = .025
nn = 10 # distribution size

#%% utils
def deform_data(X):
    Y = X.copy()
    Y[:,1] *= (Y[:,0]**2+.2)/1.2
    return Y

colors = np.array([[1,0,0],
                   [0,.8,0],
                   [0,0,1],
                   [.9, 0.5, 0]])

#%% data

X = uot.tube_data(n)
dist = dict()
for s in range(4):
    # 4 distributions
    X[nn*s:nn*(s+1), :] = deform_data(np.array([(-1)**s*0.8, (-1)**(int(s/2))*0.8])[None,:]
                                      + .1*np.random.randn(nn,2))
    dist[s] = torch.rand(nn, device=device, dtype=torch.float64)
    dist[s] /= dist[s].sum()

# graph
G, h = uot.connected_eps_graph(X)

C = torch.zeros((n,n), device=device, dtype=torch.float64)
print('Compute shortest paths...')
paths = nx.shortest_path(G)
for i in range(n):
    for j in range(i):
        C[i,j] = h*len(paths[i][j])
C += C.clone().t()

Cs = dict()
for s in range(4):
    Cs[s] = C[:,nn*s:nn*(s+1)]**2

@gif.frame
def plot_(G, X, n, dist, bary, weights, colors, i, j):
    fig = plt.figure(figsize=(10,10))
    ax=fig.add_subplot()
    uot.my_draw(G, pos=X, edge_color='k',width=80/n,
              node_size=0, alpha=.5, ax=ax)
    for s in range(4):
        plt.scatter(X[s*nn:(s+1)*nn,0], X[s*nn:(s+1)*nn,1],
                    color=colors[s], s=[1000*dist[s][z].item() for z in range(nn)], label=f'{weights[s]:.2f}')
    plt.scatter(X[:,0], X[:,1], color=colors.T@weights, s=[1000*bary[z].item() for z in range(n)])
    plt.legend()

frames=[]
for (i,ti) in enumerate(np.linspace(0,1,int(size/1.5))):
    for (j,tj) in enumerate(np.linspace(0,1,size)):
        print(f'{i+1}/{size},{j+1}/{size}')
        if i%2 == 0:
            jj = size-1-j
            tjj = 1-tj
        else:
            jj, tjj = j, tj
        weights = np.array([ti*tjj,
                            ti*(1-tjj),
                            (1-ti)*tjj,
                            (1-ti)*(1-tjj)])
        weights /= weights.sum()
        print('Sinkhorn...')
        bary = uot.barycenters(Cs, dist, weights, device=device, epsilon=epsilon,
                               n_iter=1000, same_space=False)
        frames.append(plot_(G, X, n, dist, bary, weights, colors, i, jj))

gif.save(frames, 'bary.gif', duration=150)



