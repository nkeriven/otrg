# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import torch

import utils.ot as uot
import utils.data as udata
import utils.plot as uplt

import time

import gif

plt.close('all')
np.random.seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

dogif = False
savefig = True and not dogif
stepsize_v = 3 # vertical discretization
stepsize_h = 4 # horizontal discretization

#%% plot

colors = np.array([[1,0,0],
                   [0,.8,0],
                   [0,0,1],
                   [.9, 0.5, 0]])


ndist = 30 # size of distrib
nbary = 300 # size of barycenter support
epsilon = .025

ns = [10, 750, 2000]

XX = udata.tube_data(4*ndist+nbary)

dist = dict()
for s in range(4):
    XX[ndist*s:ndist*(s+1), :] = udata.deform_data(np.array([(-1)**s*0.8, (-1)**(int(s/2))*0.6])[None,:]
                                                   + .2*(2*np.random.rand(ndist,2)-1))
    dist[s] = torch.rand(ndist, device=device, dtype=torch.float64) # weights
    dist[s] /= dist[s].sum()

# plot distributions

for _,n in enumerate(ns):
    print(n)
    YY = udata.tube_data(n)
    X = np.concatenate((XX,YY), axis=0)

    # graph
    G, h = udata.connected_eps_graph(X)

    if _==0:
        plt.figure(figsize=(10,10))
        uplt.my_draw(G, pos=X, edge_color='w', width=0, node_size=0) # to obtain same size figure
        for s in range(4):
            plt.scatter(XX[s*ndist:(s+1)*ndist,0], XX[s*ndist:(s+1)*ndist,1],
                        color=colors[s], s=[10000*dist[s][z].item() for z in range(ndist)],
                        edgecolors= 'k', linewidths=1)

        if savefig:
            plt.savefig('fig/barycenters_distributions.png', bbox_inches=0)

        # plot barycenter support
        plt.figure(figsize=(10,10))
        uplt.my_draw(G, pos=X, edge_color='w', width=0, node_size=0)
        plt.scatter(XX[4*ndist:4*ndist+nbary,0], XX[4*ndist:4*ndist+nbary,1],
                    color='k', s=40, edgecolors='k', linewidths=1)
        if savefig:
            plt.savefig('fig/barycenters_support.png', bbox_inches=0)

    # plot graph
    plt.figure(figsize=(10,10))
    uplt.my_draw(G, pos=X, edge_color='k',width=80/X.shape[0],
                 node_size=0, alpha=.5)
    if savefig:
        plt.savefig(f'fig/barycenters_graph{n}.png', bbox_inches=0)


    C = torch.zeros((n,n), device=device, dtype=torch.float64)
    print('Compute shortest paths...')
    t = time.time()
    C, SP = uot.SP_matrix(G, device=device, h=h)
    print(time.time()-t)

    Cs = dict()

    t = time.time()

    for s in range(4):
        Cs[s] = C[4*ndist:4*ndist+nbary,
                  ndist*s:ndist*(s+1)]**2
        #C, SP[s] = uot.SP_matrix(G, indices = (np.arange(4*ndist,4*ndist+nbary),
        #                                       np.arange(ndist*s,ndist*(s+1))),
        #                         device=device, h=h)
        #Cs[s] = C**2

    print(time.time()-t)

    # single barycenter
    weights = np.array([.1, .3, .2, .4])
    bary, Ps = uot.barycenters(Cs, dist, weights, device=device, epsilon=0.02,
                               n_iter=1000, same_space=False)

    plt.figure(figsize=(10,10))
    uplt.my_draw(G, pos=X, edge_color='k',width=80/X.shape[0],
                 node_size=0, alpha=.5)
    plt.scatter(X[4*ndist:4*ndist+nbary,0], X[4*ndist:4*ndist+nbary,1], color='k',
                s=10)
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
    plt.scatter(X[4*ndist:4*ndist+nbary,0], X[4*ndist:4*ndist+nbary,1], color=colors.T@weights,
                s=[10000*bary[z].item() for z in range(nbary)], edgecolors= 'k', linewidths=1)
    plt.legend(fontsize=20)
    if savefig:
        plt.savefig(f'fig/barycenters_tube{n}.png', bbox_inches=0)

"""
@gif.frame
def plot_(G, X, n, dist, bary, weights, colors, Ps):
    plt.figure(figsize=(10,10))
    uplt.my_draw(G, pos=X, edge_color='k',width=80/n,
                 node_size=0, alpha=.5)
    plt.scatter(X[4*ndist:4*ndist+nbary,0], X[4*ndist:4*ndist+nbary,1], color='k',
                s=10)
    for s in range(4):
        e_weights = uplt.compute_edge_weights_SP(G, Ps[s], SP[s], n,
                                                 indi = np.arange(4*ndist,4*ndist+nbary),
                                                 indj = np.arange(ndist*s,ndist*(s+1)),
                                                 scale = 40)
        uplt.my_draw(G, pos=X, edge_color=colors[s], width=weights[s]*np.array(e_weights),
                     node_size=0, alpha=1)
        plt.scatter(X[s*ndist:(s+1)*ndist,0], X[s*ndist:(s+1)*ndist,1],
                    color=colors[s], s=[3000*dist[s][z].item() for z in range(ndist)],
                    label=f'{weights[s]:.2f}', edgecolors= 'k', linewidths=2)
    plt.scatter(X[4*ndist:4*ndist+nbary,0], X[4*ndist:4*ndist+nbary,1], color=colors.T@weights,
                s=[5000*bary[z].item() for z in range(nbary)], edgecolors= 'k', linewidths=2)
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
                               n_iter=1000, same_space=False)
        if dogif:
            frames.append(plot_(G, X, n, dist, bary, weights, colors, Ps))
        else:
            plot_.__wrapped__(G, X, n, dist, bary, weights, colors, Ps)
            if savefig:
                plt.savefig(f'fig/tube_barycenters{i}{j}.png',
                            bbox_inches=0)

if dogif:
    gif.save(frames, 'fig/bary_tube.gif', duration=int(3000/(stepsize_v*stepsize_h)))

    """
