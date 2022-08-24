# -*- coding: utf-8 -*-


from utils.nx_pylab3d import draw3d
import numpy as np
import networkx as nx

def my_draw(G,
            node_size=20,
            node_color='b',
            width=.1,
            edge_color='gray',
            pos=None,
            vmin=None, vmax=None, **kwds):
    if type(pos) == str:
        poss = [G.nodes[i][pos] for i in G.nodes]
        pos=poss
    nx.draw(G, node_size=node_size, node_color=node_color, width=width,
            edge_color=edge_color, pos = pos,
            vmin=vmin, vmax=vmax, **kwds)

def my_draw3d(G, ax=None, fig=None,
              node_size=20,
              node_color='b',
              width=.1,
              edge_color='gray',
              pos=None,
              alpha_edge=.5,
              **kwds):
    if type(pos) == str:
        poss = [G.nodes[i][pos] for i in G.nodes]
        pos=poss
    draw3d(G, ax=ax, fig=fig, node_size=node_size, node_color=node_color, width=width,
           edge_color=edge_color, pos=pos,
           alpha_edge=alpha_edge, **kwds)

def expand_plan(P,n,N):
    PP = np.zeros((N,N))
    PP[:n, n:] = P
    PP[n:, :n] = P.T
    PP /= PP.max()
    c = .7*np.ones(N)
    c[:n] = 0
    return PP, c

def compute_edge_weights_SP(G, P, SP, n, indi, indj, scale=20):
    """ Add the weights of differents, possibly overlapping, shortest paths.
    """
    e_weights = np.zeros((n,n))
    for ii, _i in enumerate(indi):
        for jj, _j in enumerate(indj):
            p = SP[_i][_j]
            subG = G.subgraph(p)
            for e in subG.edges:
                e_weights[e[0], e[1]] += scale*P[ii,jj].item()
    e_weights += e_weights.T
    ee_weights = [e_weights[e[0], e[1]] for e in G.edges]
    return ee_weights