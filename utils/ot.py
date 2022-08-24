
import numpy as np
import torch
import networkx as nx


#%% graph dist

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

def SP_matrix(G, pos=None, indices=None, device='cpu', h=None):
    if h is None:
        h = 1
    for e in G.edges:
        if pos is not None:
            G.edges[e]['dist'] = np.linalg.norm(pos[e[0],:]-pos[e[1],:])
        else:
            G.edges[e]['dist'] = h

    if indices is None:
        n = len(G)
        C = torch.zeros((n,n), device=device, dtype=torch.float64)
        SP = nx.shortest_path(G, weight='dist')
        indi, indj = np.arange(n), np.arange(n)
    else:
        indi, indj = indices[0], indices[1]
        C = torch.zeros((len(indi),len(indj)), device=device, dtype=torch.float64)

    p = dict()
    for i, ii in enumerate(indi):
        p[ii] = dict()
        for j, jj in enumerate(indj):
            if indices is None:
                p[ii][jj] = SP[ii][jj]
            else:
                p[ii][jj] = nx.shortest_path(G, source=ii, target=jj)
            length = 0
            for v in range(len(p[ii][jj])-1):
                length += G.edges[(p[ii][jj][v], p[ii][jj][v+1])]['dist']
            C[i,j] = length

    return C, p

#%% OT

def cost_fun(f, g, alpha, beta, K, epsilon=.1):
    return f.dot(alpha) + g.dot(beta) \
        - epsilon*(np.exp(f/epsilon)*alpha).dot(K@(np.exp(g/epsilon)*beta)) \
        + epsilon # normally, not necessary at the optimum

def KL(P,PP):
    return (P*np.log(P/PP)).sum()

def sinkhorn_dual(C, alpha=None, beta=None, epsilon=.1, g_init=None,
                  n_iter=1000, K=None, eta=None, dolog=True, device='cpu'):
    """ Sinkhorn's algorithm, dual version, allowing for customized K."""

    if alpha is None:
        alpha = torch.ones(C.shape[0], device=device, dtype=torch.float64)/C.shape[0]
    if beta is None:
        beta = torch.ones(C.shape[1], device=device, dtype=torch.float64)/C.shape[1]

    # here f and g are directly divided by epsilon
    if g_init is None:
        g = torch.zeros(len(beta), device=device, dtype=torch.float64)
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

#%% barycenters

def barycenters(C, distribs, weights, epsilon=.1, n_iter=100,
                K=None, same_space=True, dolog=True,
                device='cpu'):
    """
        distribs is a dict of length S.
        If same_space, matrix C/K of size (N,N) and all distribs of size (N,).
        Else, C/K dict of matrices of size (N,N_s) and distribs of size (N_s,)
    """
    # if g_init is None:
    #     g = torch.zeros(len(beta), device=device)
    # else:
    #     g = g_init

    f, g = dict(), dict()
    for s in distribs:
        g[s] = torch.zeros(len(distribs[s]), device=device)

    if dolog: # in the log domain
        ldist = dict()
        for s in distribs: ldist[s] = torch.log(distribs[s])
        if K is None:
            if same_space:
                lK = -C/epsilon
                K = torch.exp(lK)
            else:
                lK, K = dict(), dict()
                for s in distribs:
                    lK[s] = -C[s]/epsilon
                    K[s] = torch.exp(lK[s])
        else:
            print('todo')
        for i in range(n_iter):
            la = 0
            for s in distribs:
                lKK = lK if same_space else lK[s]
                # print(lKK.shape, g[s].shape, ldist[s].shape)
                f[s] = - torch.logsumexp(lKK + (g[s]+ldist[s])[None,:], axis=1)
                la += weights[s]*torch.logsumexp(lKK + (g[s]+ldist[s])[None,:], axis=1)
            for s in distribs:
                lKK = lK if same_space else lK[s]
                g[s] = - torch.logsumexp(lKK + (f[s]+la)[:,None], axis=0)
            # if eta is not None:
            #     c = torch.log(eta)
            #     f = cutoff(f, vmax=c, vmin=-c)
            #     g = cutoff(g, vmax=c, vmin=-c)
        # P = torch.exp((f+la)[:,None]+lK+(g+lb)[None,:])
    else:
        print('todo')
        # if K is None:
        #     K = torch.exp(-C/epsilon)
        # for i in range(n_iter):
        #     f = - torch.log(K@(torch.exp(g)*beta))
        #     g = - torch.log(K.t()@(torch.exp(f)*alpha))
        #     if eta is not None:
        #         c = np.log(eta)
        #         f = cutoff(f, vmax=c, vmin=-c)
        #         g = cutoff(g, vmax=c, vmin=-c)
        # P = (torch.exp(f)*alpha)[:,None]*K*(torch.exp(g)*beta)[None,:]
    # f *= epsilon
    # g *= epsilon
    # cost = cost_fun(f.cpu(), g.cpu(), alpha.cpu(), beta.cpu(), K.cpu(), epsilon=epsilon)
    return torch.exp(la)





#%% plot

