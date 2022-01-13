
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from scipy.spatial.distance import cdist

import utils_ot as uot

#%%###############
plt.close('all')
np.random.seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
save = False

fontsize = 22
figsize = (6,5)
rc_fonts = {
    "text.usetex": True,
    "font.size": 14
}
plt.rcParams.update(rc_fonts)

# which experiment to perform
illustrate_usvt = False
stability_gamma = False # quite long
conv_curves = True # long

n_test = 10 # number of experiments to average over

# data distrib
def generate_two_circles(n, noise=.1):
    X = np.random.randn(3*n,2)
    # inner circle
    X[:n,:] = .6*X[:n,:]*((1+noise*np.random.randn(n))/(np.sqrt((X[:n,:]**2).sum(axis=1))))[:,None]
    # outer circle
    X[n:,:] = X[n:,:]/(np.sqrt((X[n:,:]**2).sum(axis=1)))[:,None]
    return X

#%% Figure 1: UVST estimator example
if illustrate_usvt:
    n = 80
    alpha = torch.ones(n, device=device, dtype=torch.float64)/n
    beta = torch.ones(2*n, device=device, dtype=torch.float64)/(2*n)
    epsilon, sigma = .01, .17
    rho = 1

    # generate data and graph
    X = generate_two_circles(n)
    G, W = uot.random_graph_similarity(X, rho = rho, mode='Gaussian',
                                       bandwidth=sigma, return_expected=True)
    W /= rho

    # sinkhorn
    C = torch.tensor(np.ones((n, 2*n)) - W[:n, n:], device=device)
    P, _, _, _ = uot.sinkhorn_dual(C, alpha, beta, epsilon=epsilon, device=device)

    # USVT: select best among several threshold value (just here)
    A = torch.tensor(nx.to_numpy_array(G), device=device)
    best_loss = np.Inf
    best_W = None
    for gamma in np.linspace(.1, .7, 25):
        What = uot.USVT(A, gamma=gamma, rho=rho, verbose=False)
        l = np.linalg.norm(What.cpu().numpy() - W, ord='fro')/n
        if l<best_loss:
            best_loss = l
            best_W = What

    # sinkhorn on USVT
    Chat = torch.ones((n, 2*n), device=device) - best_W[:n, n:]
    Phat, _, _, _ = uot.sinkhorn_dual(Chat, alpha, beta, epsilon=epsilon, device=device)

    # plot USVT
    node_size = 40
    figsize = (5,5)
    plt.figure(figsize=figsize)
    uot.my_draw(G, width=.5, pos=X,
                node_size=node_size, edge_color='k', node_color='r')
    G_c = nx.complete_graph(3*n)
    plt.figure(figsize=figsize)
    uot.my_draw(G_c, width=[W[i,j] for (i,j) in G_c.edges], pos=X,
                node_size=node_size, edge_color='k', node_color='r')
    plt.figure(figsize=figsize)
    best_W = best_W.cpu().numpy()
    uot.my_draw(G_c, width=[best_W[i,j] for (i,j) in G_c.edges], pos=X,
                node_size=node_size, edge_color='k', node_color='r')

    # plot sinkhorn
    P = P.cpu().numpy()
    Phat = Phat.cpu().numpy()
    def rect(P,n):
        PP = np.zeros((3*n,3*n))
        PP[:n, n:] = P
        PP[n:, :n] = P.T
        return PP
    PP = rect(P,n)/P.max()
    PPhat = rect(Phat,n)/Phat.max()
    c = .7*np.ones(3*n)
    c[:n] = 0

    plt.figure(figsize=figsize)
    uot.my_draw(G_c, width=[PP[i,j] for (i,j) in G_c.edges],
                pos=X, node_color=c, vmax=1,
                node_size=node_size, edge_color='k')
    plt.figure(figsize=figsize)
    uot.my_draw(G_c, width=[PPhat[i,j] for (i,j) in G_c.edges],
                pos=X, node_color=c, vmax=1,
                node_size=node_size, edge_color='k')

#%% Figure 2: stability wrt gamma

if stability_gamma:
    epsilon, sigma = .1, .2
    gammas = np.linspace(.2, .7, 10)
    ns = [100, 250, 500]
    output = np.zeros((len(gammas), 3, 2, n_test)) # n, rho, value, n_test

    # compute approximate true value: average over many realizations with true W
    true_cost = 0
    N = max(ns)
    alpha = torch.ones(N, device=device, dtype=torch.float64)/N
    beta = torch.ones(2*N, device=device, dtype=torch.float64)/(2*N)
    for t_ in range(n_test):
        X = generate_two_circles(N)
        G, W = uot.random_graph_similarity(X, rho = 1, mode='Gaussian',
                                           bandwidth=sigma, return_expected=True)
        C = torch.tensor(np.ones((N, 2*N)) - W[:N, N:], device=device)
        P, _, _, tc = uot.sinkhorn_dual(C, alpha, beta, epsilon=epsilon, device=device)
        true_cost += tc.item()/n_test

    for g_, gamma in enumerate(gammas):
        for t_ in range(n_test):
            for n_, n in enumerate(ns):
                print(f'Gamma value {g_+1}/{len(gammas)}, Num test {t_+1}/{n_test}, Graph size {n_+1}/{len(ns)}')
                alpha = torch.ones(n, device=device, dtype=torch.float64)/n
                beta = torch.ones(2*n, device=device, dtype=torch.float64)/(2*n)
                X = generate_two_circles(n)
                G, W = uot.random_graph_similarity(X, rho = 1, mode='Gaussian',
                                                   bandwidth=sigma, return_expected=True)
                A = torch.tensor(nx.to_numpy_array(G), device=device)
                What = uot.USVT(A, gamma=gamma, rho=1)
                Chat = torch.ones((n, 2*n), device=device) - What[:n, n:]
                Phat, _, _, c = uot.sinkhorn_dual(Chat, alpha, beta,
                                                  epsilon=epsilon, n_iter=1000,
                                                  device=device)
                output[g_, n_, 0, t_] += np.abs(true_cost-c.item())/true_cost
                output[g_, n_, 1, t_] += np.linalg.norm(W - What.cpu().numpy(),
                                                        ord='fro')/n

    output_m = output.mean(axis=-1)
    plt.figure(figsize=figsize)
    for _ in range(3):
        plt.semilogy(gammas, output_m[:,_,0], linewidth=4)
    plt.xlabel(r'$\gamma$', fontsize=fontsize)
    plt.ylim(top=.08)
    plt.grid(color='0.8', linestyle='--', which = 'both')
    plt.legend([r'$n=100$', r'$n=250$', r'$n=500$'], fontsize=fontsize,
               ncol=2)
    plt.tight_layout()
    plt.savefig('gaussian_conv_gamma_Wass.pdf', bbox_inches=0)

    plt.figure(figsize=figsize)
    for _ in range(3):
        plt.semilogy(gammas, output_m[:,_,1], linewidth=4)
    plt.xlabel(r'$\gamma$', fontsize=fontsize)
    plt.grid(color='0.8', linestyle='--', which = 'both')
    plt.tight_layout()
    plt.savefig('gaussian_conv_gamma_usvt.pdf', bbox_inches=0)

#%% Figure 3: convergence

if conv_curves:

    ### USVT
    epsilon, sigma = 0.05, 0.17
    ns = np.logspace(2.3,3.3,7).astype(int)
    output = np.zeros((len(ns), 3, 3, n_test)) # n, rho, value, n_test

    for n_, n in enumerate(ns):
        for t_ in range(n_test):
            alpha = torch.ones(n, device=device, dtype=torch.float64)/n
            beta = torch.ones(2*n, device=device, dtype=torch.float64)/(2*n)
            X = generate_two_circles(n)
            G, W = uot.random_graph_similarity(X, rho = 1, mode='Gaussian',
                                               bandwidth=sigma, return_expected=True)
            C = torch.tensor(np.ones((n, 2*n)) - W[:n, n:], device=device)
            P, _, _, tc = uot.sinkhorn_dual(C, alpha, beta, epsilon=epsilon, device=device)
            for r_, rho in enumerate([1, 1/n**(1/6), 1/n**(1/3)]):
                print(f'Graph size {n_+1}/{len(ns)}, Num test {t_+1}/{n_test}, Sparsity {r_+1}/3')
                
                G, W = uot.random_graph_similarity(X, rho = rho, mode='Gaussian',
                                                   bandwidth=sigma, return_expected=True)
                W /= rho
                A = torch.tensor(nx.to_numpy_array(G), device=device)
                What = uot.USVT(A, gamma=.5, rho=rho)
                Chat = torch.ones((n, 2*n), device=device) - What[:n, n:]
                Phat, _, _, c = uot.sinkhorn_dual(Chat, alpha, beta,
                                                  epsilon=epsilon, n_iter=1000,
                                                  device=device)
                output[n_, r_, 0, t_] += np.abs(tc-c.item())/tc
                output[n_, r_, 1, t_] += np.linalg.norm(W - What.cpu().numpy(),
                                                        ord='fro')/n
                output[n_, r_, 2, t_] += uot.KL(Phat.cpu().numpy(), P.cpu().numpy())

    output_m = output.mean(axis=-1)

    ### fast rate
    sigma = .5
    epsilon = 2*sigma**2
    eta = np.exp(4/epsilon)
    output_fast = np.zeros((len(ns), 3, 2, n_test)) # n, rho, value, n_test

    # approximate true cost
    # true_cost = 0
    # N = ns.max()
    # alpha = torch.ones(N, device=device, dtype=torch.float64)/N
    # beta = torch.ones(2*N, device=device, dtype=torch.float64)/(2*N)
    # for t_ in range(n_test):
    #     X = generate_two_circles(N)
    #     C = torch.tensor(cdist(X[:N,:], X[N:,:], 'sqeuclidean'), device=device)
    #     P, _, _, tc = uot.sinkhorn_dual(C, alpha, beta, epsilon=epsilon, device=device)
    #     true_cost += tc.item()/n_test

    for n_, n in enumerate(ns):
        for t_ in range(n_test):
            alpha = torch.ones(n, device=device, dtype=torch.float64)/n
            beta = torch.ones(2*n, device=device, dtype=torch.float64)/(2*n)
            X = generate_two_circles(n)
            C = torch.tensor(cdist(X[:n,:], X[n:,:], 'sqeuclidean'), device=device)
            P, _, _, tc = uot.sinkhorn_dual(C, alpha, beta, epsilon=epsilon, device=device)
            for r_, rho in enumerate([1, 1/n**(1/6), 1/n**(1/3)]):
                print(f'Graph size {n_+1}/{len(ns)}, Num test {t_+1}/{n_test}, Sparsity {r_+1}/3')
                alpha = torch.ones(n, device=device, dtype=torch.float64)/n
                beta = torch.ones(2*n, device=device, dtype=torch.float64)/(2*n)
                X = generate_two_circles(n)
                G, W = uot.random_graph_similarity(X, rho = rho, mode='Gaussian',
                                                   bandwidth=sigma, return_expected=True)
                W /= rho
                Khat = torch.tensor(nx.to_numpy_array(G), device=device)[:n, n:]/rho
                Phat, _, _, c = uot.sinkhorn_dual(None, alpha, beta,
                                                  epsilon=epsilon, n_iter=1000,
                                                  device=device, K=Khat,
                                                  eta=eta, dolog=False)
                output_fast[n_, r_, 0, t_] += np.abs(tc-c.item())/tc
                output_fast[n_, r_, 1, t_] += np.linalg.norm(Khat.cpu().numpy() - W[:n,n:],
                                                        ord=2)/n

    output_m_fast = output_fast.mean(axis=-1)

    ### plot
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    lab = [r'$\rho = 1$', r'$\rho = n^{-1/6}$', r'$\rho=n^{-1/3}$']

    plt.figure(figsize=figsize)
    for _ in range(3):
        plt.loglog(ns, output_m[:,_,0], linewidth=4, color=colors[_], label = lab[_])
        plt.loglog(ns, output_m_fast[:,_,0], '--', linewidth=4, c=colors[_])

    plt.xlabel(r'$n$', fontsize=fontsize)
    plt.grid(color='0.8', linestyle='--', which = 'both')
    plt.tight_layout()
    plt.savefig('Wass.pdf', bbox_inches=0)

    plt.figure(figsize=figsize)
    for _ in range(3):
        plt.loglog(ns, output_m[:,_,1], linewidth=4, color=colors[_], label = lab[_])
        plt.loglog(ns, output_m_fast[:,_,1], '--', linewidth=4, c=colors[_])

    plt.xlabel(r'$n$', fontsize=fontsize)
    plt.grid(color='0.8', linestyle='--', which = 'both')
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('spectral.pdf', bbox_inches=0)
    
    plt.figure(figsize=figsize)
    for _ in range(3):
        plt.loglog(ns, output_m[:,_,2], linewidth=4, color=colors[_], label = lab[_])

    plt.xlabel(r'$n$', fontsize=fontsize)
    plt.grid(color='0.8', linestyle='--', which = 'both')
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('KL.pdf', bbox_inches=0)

