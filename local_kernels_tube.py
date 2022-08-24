# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 10:21:22 2021

@author: nicol
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import utils.ot as uot
import utils.data as udata
import utils.plot as uplt
import torch

from scipy import stats
from scipy.integrate import quad

import gif

# init
plt.close('all')
np.random.seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# plot parameters
fontsize = 20
figsize = (6,4)
rc_fonts = {"text.usetex": True, "font.size": 14}
plt.rcParams.update(rc_fonts)

# illustrate the OT plan for increasing n
illustrate_conv_SP = True
# create a gif animating the OT plan (a bit long)
illustrate_gif_OT_plan = illustrate_conv_SP and False

# do I save the figures?
savefig = False

n_test = 10 # number of experiments to average over

#%% conv TP

epsilon = .05
nn = 30
posA, posB = 2*np.random.rand(nn)-1, 2*np.random.rand(nn)-1
# posA, posB = np.linspace(-1,1,nn), np.linspace(-1,1,nn)
alpha = torch.rand(nn, device=device, dtype=torch.float64)
beta = torch.rand(nn, device=device, dtype=torch.float64)
alpha /= alpha.sum()
beta /= beta.sum()
if illustrate_conv_SP:
    for n in [300, 1000, 3000]:
        # data
        X = udata.tube_data(n)
        X[:nn, 0] = -1
        X[:nn, 1] = posA
        X[nn:2*nn, 0] = 1
        X[nn:2*nn, 1] = posB

        # graph
        G, h = udata.connected_eps_graph(X)

        C = torch.zeros((nn,nn), device=device, dtype=torch.float64)

        C, p = uot.SP_matrix(G, indices=(np.arange(nn), np.arange(nn,2*nn)),
                             h=h, device=device)

        P, _, _, c = uot.sinkhorn_dual(C, alpha, beta,
                                       epsilon=epsilon, n_iter=1000,
                                       device=device)
        P = P.cpu().numpy()
        ## plot
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot()
        uplt.my_draw(G, pos=X, edge_color='g',width=100/n,
                      node_size=0, alpha=.5, ax=ax)
        # distributions
        for i in range(nn):
            plt.scatter(X[i,0], X[i,1], color='b', s=2000*alpha[i].item())
            plt.scatter(X[nn+i,0], X[nn+i,1], color='r', s=2000*beta[i].item())

        e_weights = uplt.compute_edge_weights_SP(G, P, p, n,
                                                 np.arange(nn), np.arange(nn, 2*nn))

        uplt.my_draw(G, pos=X, edge_color='k', width=e_weights,
                    node_size=0, alpha=1, ax=ax)
        if savefig:
            plt.savefig(f'local_tube_OTplan_{n}.png',
                        bbox_inches= 0, transparent=True)

if illustrate_gif_OT_plan:
    @gif.frame
    def draw_transport_plan(X, G, nn, paths, t, P):
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot()
        uplt.my_draw(G, pos=X, edge_color='k',width=.05,
                      node_size=0, alpha=.5, ax=ax)
        t_plan = dict()
        for i in range(nn):
            plt.scatter(X[i,0], X[i,1], color='b', s=2000*alpha[i].item())
            plt.scatter(X[nn+i,0], X[nn+i,1], color='r', s=2000*beta[i].item())
        for i in range(nn):
            for j in range(nn):
                pp = p[i][nn+j]
                # gdraw.my_draw(G.subgraph(pp), pos=X, edge_color='g',width=100*P[i,j].item(),
                #               node_size=0, alpha=1, ax=ax)
                node = pp[int(t*(len(pp)-1))]
                if node in t_plan:
                    t_plan[node] += P[i,j]
                else:
                    t_plan[node] = P[i,j]
        for node in t_plan:
            plt.scatter(X[node, 0], X[node, 1], color=np.array([t,0,1-t]),
                        s=2000*t_plan[node])

    frames = []
    for t in np.linspace(0,1,100):
        frames.append(draw_transport_plan(X,G,nn,p,t,P))
    gif.save(frames, 'tube_OTplan.gif', duration=90)


