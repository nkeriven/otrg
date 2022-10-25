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
figpath = 'fig/'
savefig = True
stepsize_v = 8 # vertical discretization
stepsize_h = 10 # horizontal discretization

#%% plot

colors = uplt.mycolor()
n = 2000
epsilon = .025
ndist = 20
X = udata.tube_data(n)

# distributions
dist = dict()
for s in range(4):
    X[ndist*s:ndist*(s+1), :] = udata.deform_data(np.array([(-1)**s*0.8, (-1)**(int(s/2))*0.6])[None,:]
                                                  + .2*(2*np.random.rand(ndist,2)-1))
    dist[s] = torch.rand(ndist, device=device, dtype=torch.float64)
    dist[s] /= dist[s].sum()

# graph
G, h = udata.connected_eps_graph(X)

C = torch.zeros((n,n), device=device, dtype=torch.float64)
print('Compute shortest paths...')
C, SP = uot.SP_matrix(G, device=device, h=h)

Cs = dict()
for s in range(4):
    Cs[s] = C[4*ndist:,ndist*s:ndist*(s+1)]**2

@gif.frame
def plot_(G, X, n, Ps, dist, bary, weights):
    plt.figure(figsize=(10,10))
    uplt.my_draw(G, pos=X, edge_color='k',width=80/X.shape[0],
                 node_size=0, alpha=.5)
    # draw transport plan
    for s in range(4):
        e_weights = uplt.compute_edge_weights_SP(G, Ps[s], SP, X.shape[0],
                                                 indi = np.arange(4*ndist,n),
                                                 indj = np.arange(ndist*s,ndist*(s+1)),
                                                 scale = 80)
        uplt.my_draw(G, pos=X, edge_color=colors[s], width=weights[s]*np.array(e_weights),
                     node_size=0, alpha=1)
        plt.scatter(X[s*ndist:(s+1)*ndist,0], X[s*ndist:(s+1)*ndist,1],
                    color=colors[s], s=[10000*dist[s][z].item() for z in range(ndist)],
                    label=f'{weights[s]:.2f}', edgecolors= 'k', linewidths=1)
    # draw barycenter weights
    plt.scatter(X[4*ndist:,0], X[4*ndist:,1], color=colors.T@weights,
                s=[10000*bary[z].item() for z in range(len(bary))], edgecolors= 'k', linewidths=1)
    plt.legend(fontsize=20)


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
        bary, Ps = uot.barycenters(Cs, dist, weights, device=device, epsilon=epsilon,
                                   n_iter=300, same_space=False)
        if savefig:
            frames.append(plot_(G, X, n, Ps, dist, bary, weights))

if savefig:
    gif.save(frames, figpath + 'bary_tube.gif', duration=int(10000/(stepsize_v*stepsize_h)))

