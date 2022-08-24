
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import utils.ot as uot
import utils.data as udata
import utils.plot as uplt
import torch

# init
plt.close('all')
np.random.seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# plot parameters
fontsize = 20
figsize = (6,4)
rc_fonts = {"text.usetex": True, "font.size": 14}
plt.rcParams.update(rc_fonts)

# which experiment to perform
# Illustrate the convergence of the shortest paths on the sphere
illustration_conv_SP = True
# Illustrate a transport plan on the sphere
illustration_OT_sphere = True
# Compute the convergence curves. Can be quite long
curve_conv_wrt_n = True

n_test = 10 # number of experiments to average over
savefig = False # do I save the figures?

#%% Fig 4.1 (top)

if illustration_conv_SP:
    for n in [200, 500, 6000]:
        X = udata.generate_sphere(n)

        # two fixed points
        X[-2,:] = [1,0,0]
        X[-1,:] = [-.5,1,0]/np.sqrt(.5**2+1)
        # true distance
        c = np.arccos(X[-2,:].dot(X[-1,:]))

        # true path
        N = 100
        t = np.linspace(0,1,N)
        Y = t[:,None] * X[-2,:] + (1-t)[:,None] * X[-1,:]
        Y = Y/np.sqrt((Y**2).sum(axis=1))[:,None]
        lG = nx.path_graph(N)

        G, h = udata.connected_eps_graph(X)

        p = nx.algorithms.shortest_path(G, source=n-2, target=n-1)

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')
        uplt.my_draw3d(G, pos=X, edge_color='k',width=.5*np.sqrt(200/n),
                      node_size=0, alpha_edge=.3, alpha=.5, ax=ax)
        uplt.my_draw3d(G.subgraph(p), pos=X*1.02, edge_color='r',width=5, node_color='r',
                      alpha_edge=1, node_size=50, alpha=1, ax=ax)
        uplt.my_draw3d(lG, pos=Y*1.01, edge_color='g',width=5, node_color='g',
                      alpha_edge=1, node_size=5, alpha=1, ax=ax)
        ax=plt.gca()
        ax.view_init(azim=66, elev=30)
        if savefig:
            plt.savefig(f'shortest_path_sphere_{n}.png', bbox_inches=0)

#%% generate data for transport plan and convergence

if illustration_OT_sphere or curve_conv_wrt_n:
    epsilon, bw = .01, .2
    # points
    dist_size = 50
    Y1 = udata.normalize(.4*(2*np.random.rand(dist_size, 3)-1) + np.array([1,0,0])[None,:])
    Y2 = np.zeros((dist_size, 3))
    Y2[:, 1:3] = udata.normalize(2*np.random.rand(dist_size, 2)-1)
    Y2 = udata.normalize(Y2 + np.array([.5,0,0])[None,:])

    # compute true OT cost
    C = torch.tensor(np.arccos(Y1 @ Y2.T), device=device)
    P, _, _, true_cost = uot.sinkhorn_dual(C, epsilon=epsilon, device=device)
    true_cost = true_cost.item()

#%% Illustration of OT plan



if illustration_OT_sphere:
    ndraw=1000
    X = udata.generate_sphere(ndraw)
    h = ndraw**(-1/4)
    G, h = udata.connected_eps_graph(X)

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    uplt.my_draw3d(G, pos=X, edge_color='k',width=.2,
                  node_size=0, alpha_edge=.3, alpha=.5, ax=ax)

    PP, c = uplt.expand_plan(P.cpu().numpy(), dist_size, 2*dist_size)
    G_c = nx.complete_graph(2*dist_size)
    uplt.my_draw3d(G_c, width=[PP[i,j] for (i,j) in G_c.edges],
                  pos=np.concatenate((Y1,Y2), axis=0), node_color=c, vmax=1,
                  node_size=50, edge_color='k', ax=ax)
    ax=plt.gca()
    ax.view_init(azim=22, elev=24)
    if savefig:
        plt.savefig('OT_plan_sphere.png', bbox_inches=0)

#%% convergence wrt n

if curve_conv_wrt_n:
    ns = np.logspace(1.5,3.6,7).astype(int)
    output = np.zeros((len(ns), 3, n_test)) # n, rho, value, n_test
    for n_, n in enumerate(ns):
        for t_ in range(n_test):
            print(f'Graph size {n_+1}/{len(ns)}, num test {t_+1}/{n_test}')
            X = udata.generate_sphere(n)
            X = np.concatenate((Y1,Y2,X), axis=0)

            G, eps = udata.connected_eps_graph(X)

            Chat = np.zeros((dist_size, dist_size))
            for i in range(dist_size):
                for j in range(dist_size):
                    Chat[i,j] = eps*len(nx.algorithms.shortest_path(G, source=i, target= dist_size+j))
            Chat = torch.tensor(Chat, device=device)
            Phat, _, _, c = uot.sinkhorn_dual(Chat,
                                              epsilon=epsilon, n_iter=1000,
                                              device=device)
            output[n_, 0, t_] += np.abs(true_cost-c.item())/true_cost
            output[n_, 1, t_] += np.abs((C-Chat).cpu().numpy()).max()
            output[n_, 2, t_] += uot.KL(Phat.cpu().numpy(), P.cpu().numpy())

    output_m = output.mean(axis=-1)
    plt.figure(figsize=figsize)
    for _ in range(3):
        plt.loglog(ns, output_m[:,_], linewidth=4)
    plt.xlabel(r'$N$', fontsize=fontsize)
    plt.grid(color='0.8', linestyle='--', which = 'both')
    plt.legend([r'OT error', r'$\|\hat{C} - C\|_\infty$',
                r'$KL(P^C, P^{\hat{C}})$'], fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('shortest.pdf', bbox_inches=0)