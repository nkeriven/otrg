# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import torch

import utils.ot as uot
import utils.data as udata
import utils.plot as uplt

import gif

plt.close('all')
np.random.seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

dogif = True
savefig = True and not dogif
stepsize_v = 3 # vertical discretization
stepsize_h = 4 # horizontal discretization

#%% plot

colors = np.array([[1,0,0],
                   [0,.8,0],
                   [0,0,1],
                   [.9, 0.5, 0]])

n = 1000
epsilon = .025
nn = 10
X = udata.tube_data(n)
dist = dict()
for s in range(4):
    X[nn*s:nn*(s+1), :] = udata.deform_data(np.array([(-1)**s*0.8, (-1)**(int(s/2))*0.8])[None,:]
                                            + .1*np.random.randn(nn,2))
    dist[s] = torch.rand(nn, device=device, dtype=torch.float64)
    dist[s] /= dist[s].sum()

# graph
G, h = udata.connected_eps_graph(X)

C = torch.zeros((n,n), device=device, dtype=torch.float64)
print('Compute shortest paths...')
C, SP = uot.SP_matrix(G, device=device, h=h)

Cs = dict()
for s in range(4):
    Cs[s] = C[:,nn*s:nn*(s+1)]**2

@gif.frame
def plot_(G, X, n, dist, bary, weights, colors):
    fig = plt.figure(figsize=(10,10))
    ax=fig.add_subplot()
    uplt.my_draw(G, pos=X, edge_color='k',width=80/n,
              node_size=0, alpha=.5, ax=ax)
    for s in range(4):
        plt.scatter(X[s*nn:(s+1)*nn,0], X[s*nn:(s+1)*nn,1],
                    color=colors[s], s=[1000*dist[s][z].item() for z in range(nn)], label=f'{weights[s]:.2f}')

    plt.scatter(X[:,0], X[:,1], color=colors.T@weights, s=[1000*bary[z].item() for z in range(n)])
    plt.legend()


frames=[]
for (i,ti) in enumerate(np.linspace(0,1,stepsize_v)):
    for (j,tj) in enumerate(np.linspace(0,1,stepsize_h)):
        print(f'{i+1}/{stepsize_v}, {j+1}/{stepsize_h}')
        if i%2 == 0:
            jj = stepsize_h-1-j
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
        if dogif:
            frames.append(plot_(G, X, n, dist, bary, weights, colors))
        else:
            plot_.__wrapped__(G, X, n, dist, bary, weights, colors)
            if savefig:
                plt.savefig(f'fig/tube_barycenters{i}{j}.png',
                            bbox_inches=0)

if dogif:
    gif.save(frames, 'fig/bary_tube.gif', duration=int(3000/(stepsize_v*stepsize_h)))

