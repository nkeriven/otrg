
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from numba import jit
from nx_pylab3d import draw3d

import torch

#%%###############
plt.close('all')
np.random.seed(0)

#%% OT

def cutoff(A, vmin = 0, vmax = 1):
    B = A.clone()
    B[B > vmax] = vmax
    B[B < vmin] = vmin
    return B

def USVT(A, gamma=.5, rho=1, cut=True, vmin=0, vmax=1, verbose=False):
    n = A.shape[0]
    s, v = torch.linalg.eigh(A)
    if verbose:
        print((s < gamma*np.sqrt(rho*n)).sum()/n, ' percent of eig are suppressed')
    s[s < gamma*np.sqrt(rho*n)] = 0
    Ahat = (v * s) @ v.t() / rho
    if cut:
        return cutoff(Ahat, vmin, vmax)
    return Ahat

def cost_fun(f, g, alpha, beta, K, epsilon=.1):
    return f.dot(alpha) + g.dot(beta) \
        - epsilon*(np.exp(f/epsilon)*alpha).dot(K@(np.exp(g/epsilon)*beta)) \
        + epsilon # normally, not necessary at the optimum

def KL(P,PP):
    return (P*np.log(P/PP)).sum()

def sinkhorn_dual(C, alpha, beta, epsilon=.1, g_init=None,
                  n_iter=1000, K=None, eta=None, dolog=True, device='cpu'):
    """ Sinkhorn's algorithm, dual version, allowing for customized K."""

    # here f and g are directly divided by epsilon
    if g_init is None:
        g = torch.zeros(len(beta), device=device)
    else:
        g = g_init

    if dolog: # in the log domain
        la, lb = torch.log(alpha), torch.log(beta)
        if K is None:
            lK = -C/epsilon
            K = torch.exp(lK)
        else:
            lK = torch.log(K)
        for i in range(n_iter):
            f = - torch.logsumexp(lK + (g+lb)[None,:], axis=1)
            g = - torch.logsumexp(lK + (f+la)[:,None], axis=0)
            if eta is not None:
                c = torch.log(eta)
                f = cutoff(f, vmax=c, vmin=-c)
                g = cutoff(g, vmax=c, vmin=-c)
        P = torch.exp((f+la)[:,None]+lK+(g+lb)[None,:])
    else:
        if K is None:
            K = torch.exp(-C/epsilon)
        for i in range(n_iter):
            f = - torch.log(K@(torch.exp(g)*beta))
            g = - torch.log(K.t()@(torch.exp(f)*alpha))
            if eta is not None:
                c = np.log(eta)
                f = cutoff(f, vmax=c, vmin=-c)
                g = cutoff(g, vmax=c, vmin=-c)
        P = (torch.exp(f)*alpha)[:,None]*K*(torch.exp(g)*beta)[None,:]
    f *= epsilon
    g *= epsilon
    cost = cost_fun(f.cpu(), g.cpu(), alpha.cpu(), beta.cpu(), K.cpu(), epsilon=epsilon)
    return P, f, g, cost

def random_graph_similarity(X, rho=1, mode="Gaussian", bandwidth=1,
                            return_expected = False):
    """RG model with similarity kernel that only depend on the distance
    between latent variable.

    Input
        X           : n*d latent variables
        rho         : sparsity level. Default 1
        mode        : "Gaussian" or "epsilon_graph"
        bandwidth   : bandwidth of kernel default 1
        return_expected : bool, return full expected matrix W = E(A)
    Output
        G       : NetworkX Graph
        W       : if return_expected = True. W=E(A)
    """

    n = X.shape[0]
    G = nx.empty_graph(n)

    if n>5000:
        print(f'Generate random graph with {n} nodes... (can be long)')

    for i in range(n):
        G.nodes[i]['latent'] = X[i,:]

    # generate edges
    if mode == "Gaussian":
        edgelist = generate_edges_gaussian(X, bandwidth, rho)
        G.add_edges_from(edgelist)
        if return_expected:
            W = rho * np.exp(-squareform(pdist(X, 'sqeuclidean'))
                             /(2*bandwidth**2))
            np.fill_diagonal(W,0)
    elif mode == "epsilon_graph":
        edgelist = generate_edges_epsilon_graph(X, bandwidth, rho)
        G.add_edges_from(edgelist)
        if return_expected:
            W = rho * (squareform(pdist(X, 'sqeuclidean'))<bandwidth**2)
            np.fill_diagonal(W,0)

    if return_expected:
        return G, W
    return G

@jit(nopython=True)
def generate_edges_gaussian(X,sigma,alpha):
    ret = []
    for i in range(X.shape[0]):
        for j in range(i):
            vi, vj = X[i,:], X[j,:]
            if np.random.rand() < alpha*np.exp(-((vi-vj)**2).sum()
                                               /(2*sigma**2)):
                ret.append((i,j))
    return ret

@jit(nopython=True)
def generate_edges_epsilon_graph(X,epsilon,alpha):
    ret = []
    for i in range(X.shape[0]):
        for j in range(i):
            vi, vj = X[i,:], X[j,:]
            if np.random.rand() < alpha*(((vi-vj)**2).sum()<epsilon**2):
                ret.append((i,j))
    return ret

def connected_eps_graph(X, h=None):
    n = X.shape[0]
    if h is None:
        h = 1.3*n**(-1/3)
    G = random_graph_similarity(X, mode='epsilon_graph', bandwidth=h)
    while not nx.is_connected(G):
        h *= 1.05
        G = random_graph_similarity(X, mode='epsilon_graph', bandwidth=h)
    return G, h

#%% plot

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

def rect(P,n,N):
    PP = np.zeros((N,N))
    PP[:n, n:] = P
    PP[n:, :n] = P.T
    PP = .5*PP
    PP /= PP.max()
    c = .7*np.ones(N)
    c[:n] = 0
    return PP, c