# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import torch

import utils.ot as uot
import utils.data as udata
import utils.plot as uplt

plt.close('all')
np.random.seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

savefig = True

###

plt.close('all')
np.random.seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

figpath = 'fig/'

ndist = 30 # size of distrib
nbary = 300 # size of barycenter support
epsilon = .025

ns = [10, 750, 2000] # number of additional points

XX = udata.tube_data(4*ndist+nbary)

# distributions
dist = dict()
for s in range(4):
    XX[ndist*s:ndist*(s+1), :] = udata.deform_data(np.array([(-1)**s*0.8, (-1)**(int(s/2))*0.6])[None,:]
                                                   + .2*(2*np.random.rand(ndist,2)-1))
    dist[s] = torch.rand(ndist, device=device, dtype=torch.float64) # weights
    dist[s] /= dist[s].sum()

weights = np.array([.1, .3, .2, .4])
colors = uplt.mycolor()
for _,n in enumerate(ns):
    # additional points
    YY = udata.tube_data(n)
    X = np.concatenate((XX,YY), axis=0)

    # graph
    G, h = udata.connected_eps_graph(X)

    # plot distribution and barycenter support
    if _==0:
        plt.figure(figsize=(10,10))
        uplt.my_draw(G, pos=X, edge_color='w', width=0, node_size=0) # to obtain same size figure
        for s in range(4):
            plt.scatter(XX[s*ndist:(s+1)*ndist,0], XX[s*ndist:(s+1)*ndist,1],
                        color=colors[s], s=[10000*dist[s][z].item() for z in range(ndist)],
                        edgecolors= 'k', linewidths=1)

        if savefig:
            plt.savefig(figpath+'barycenters_distributions.png', bbox_inches=0)

        # plot barycenter support
        plt.figure(figsize=(10,10))
        uplt.my_draw(G, pos=X, edge_color='w', width=0, node_size=0)
        plt.scatter(XX[4*ndist:4*ndist+nbary,0], XX[4*ndist:4*ndist+nbary,1],
                    color='k', s=40, edgecolors='k', linewidths=1)
        if savefig:
            plt.savefig(figpath+'barycenters_support.png', bbox_inches=0)

    # plot graph structure alone
    plt.figure(figsize=(10,10))
    uplt.my_draw(G, pos=X, edge_color='k',width=80/X.shape[0],
                 node_size=0, alpha=.5)
    if savefig:
        plt.savefig(figpath+f'barycenters_graph{n}.png', bbox_inches=0)

    # compute shortest path
    C = torch.zeros((n,n), device=device, dtype=torch.float64)
    print('Compute shortest paths...')
    C, SP = uot.SP_matrix(G, device=device, h=h)

    Cs = dict()
    for s in range(4):
        Cs[s] = C[4*ndist:4*ndist+nbary,
                  ndist*s:ndist*(s+1)]**2

    # barycenter computation
    bary, Ps = uot.barycenters(Cs, dist, weights, device=device, epsilon=0.02,
                               n_iter=1000, same_space=False)

    ###### draw graph
    plt.figure(figsize=(10,10))
    uplt.my_draw(G, pos=X, edge_color='k',width=80/X.shape[0],
                 node_size=0, alpha=.5)
    # draw barycenter support
    plt.scatter(X[4*ndist:4*ndist+nbary,0], X[4*ndist:4*ndist+nbary,1],
                color='k', s=10)
    # draw transport plan
    for s in range(4):
        e_weights = uplt.compute_edge_weights_SP(G, Ps[s], SP, X.shape[0],
                                                 indi = np.arange(4*ndist,4*ndist+nbary),
                                                 indj = np.arange(ndist*s,ndist*(s+1)),
                                                 scale = 100)
        uplt.my_draw(G, pos=X, edge_color=colors[s], width=weights[s]*np.array(e_weights),
                     node_size=0, alpha=1)
        plt.scatter(X[s*ndist:(s+1)*ndist,0], X[s*ndist:(s+1)*ndist,1],
                    color=colors[s], s=[10000*dist[s][z].item() for z in range(ndist)],
                    label=f'{weights[s]:.2f}', edgecolors= 'k', linewidths=1)
    # draw barycenter weights
    plt.scatter(X[4*ndist:4*ndist+nbary,0], X[4*ndist:4*ndist+nbary,1], color=colors.T@weights,
                s=[10000*bary[z].item() for z in range(nbary)], edgecolors= 'k', linewidths=1)
    plt.legend(fontsize=20)
    if savefig:
        plt.savefig(figpath + f'barycenters_tube{n}.png', bbox_inches=0)

