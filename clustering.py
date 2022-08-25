
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from scipy.stats.stats import pearsonr

import utils.ot as uot
import utils.data as udata
import utils.plot as uplt

# init
plt.close('all')
np.random.seed(0)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# do I save the figures?
savefig = True

n_test = 10 # number of experiments to average over

#%% data distrib

def OT_USVT_dist(G, node_list, node_listC=None, epsilon=.01, gamma=.5, Wusvt=None):
    n = len(G)
    s = len(node_list)
    t = n-s
    if node_listC is None:
        node_listC = [i for i in G.nodes if i not in node_list]

    # USVT
    if Wusvt is None:
        A = torch.tensor(nx.to_numpy_array(G), device=device)
        Wusvt = uot.USVT(A, gamma=gamma, rho=rho, verbose=False)
    Chat = torch.ones((s, t), device=device) - Wusvt[np.ix_(node_list, node_listC)]

    _, _, _, cost = uot.sinkhorn_dual(Chat, device=device)

    return cost

def array2comm(C):
    return np.where(C==-1)[0], np.where(C==1)[0]

def noisy_community(C,p=0):
    return C*(1-2*(np.random.rand(len(C))<p))

# def trans_community(C, CC, p = .5):
#     return C*(1-np.abs(C-CC)*(np.random.rand(len(C))<p))

def plot_comm(G, X, C):
    n = len(G)
    c = .7*np.ones(n)
    c[np.where(C==-1)[0]] = 0
    uplt.my_draw(G, width=.1,
                pos=X, node_color=c, vmax=1,
                node_size=30, edge_color='k')


#%% UVST estimator example
n = 300
epsilon = .01
rho = .5

ns = np.linspace(0,.5,15)
outputs = np.zeros((2, n_test, 6))
type_data = ['circle', 'GMM']
for data_ in range(2):
    for test_ in range(n_test):
        if type_data[data_] == 'circle':
            X = udata.generate_two_circles(n, noise=.25, noise_out=.1)
            sigma = .2
        else:
            X = udata.GMM(n, shift = 1)
            sigma = .8
        G, W = udata.random_graph_similarity(X, rho = rho, mode='Gaussian',
                                           bandwidth=sigma, return_expected=True)
        trueC = np.ones(X.shape[0])
        trueC[:n] = -1
        # badC = np.random.choice([-1,1], size=X.shape[0])

        A = torch.tensor(nx.to_numpy_array(G), device=device)
        Wusvt = uot.USVT(A, gamma=.5, rho=rho, verbose=False)
        ot = []
        cdct = []
        ncut = []
        cov = []
        perf = []
        mod = []
        for nlevel in ns:
            Sn, Tn = array2comm(noisy_community(trueC, p=nlevel))
            # Sn, Tn = array2comm(trans_community(trueC, badC, p=nlevel))
            ot.append(OT_USVT_dist(G, Sn, Tn, Wusvt=Wusvt))
            cdct.append(nx.cuts.conductance(G,Sn))
            ncut.append(nx.cuts.normalized_cut_size(G,Sn))
            cov_, perf_ = nx.community.partition_quality(G,(Sn,Tn))
            cov.append(cov_)
            perf.append(perf_)
            mod.append(nx.community.modularity(G,(Sn,Tn)))
        outputs[data_, test_, 0] = np.abs(pearsonr(ns, ot)[0])
        outputs[data_, test_, 1] = np.abs(pearsonr(ns, cdct)[0])
        outputs[data_, test_, 2] = np.abs(pearsonr(ns, ncut)[0])
        outputs[data_, test_, 3] = np.abs(pearsonr(ns, cov)[0])
        outputs[data_, test_, 4] = np.abs(pearsonr(ns, perf)[0])
        outputs[data_, test_, 5] = np.abs(pearsonr(ns, mod)[0])

    plt.figure(figsize=(5,5))
    plot_comm(G,X,trueC)
    if savefig:
        plt.savefig(f'fig/comm_graph_{type_data[data_]}_0.png',
                    bbox_inches=0)

    plt.figure(figsize=(5,5))
    plot_comm(G,X,noisy_community(trueC, p=.1))
    if savefig:
        plt.savefig(f'fig/comm_graph_{type_data[data_]}_1.png',
                    bbox_inches=0)

    plt.figure(figsize=(5,5))
    plot_comm(G,X,noisy_community(trueC, p=.5))
    if savefig:
        plt.savefig(f'fig/comm_graph_{type_data[data_]}_2.png',
                    bbox_inches=0)

outputs_ = np.mean(outputs, axis=1)

print(outputs_)

print(f'Circle:\n \
      OT {outputs_[0,0]}\n \
      Cond {outputs_[0,1]}\n \
      Cut {outputs_[0,2]}\n \
      Cov {outputs_[0,3]}\n \
      Perf {outputs_[0,4]}\n \
      Mod {outputs_[0,5]}\n \
      GMM:\n \
      OT {outputs_[1,0]}\n \
      Cond {outputs_[1,1]}\n \
      Cut {outputs_[1,2]}\n \
      Cov {outputs_[1,3]}\n \
      Perf {outputs_[1,4]}\n \
      Mod {outputs_[1,5]}')

