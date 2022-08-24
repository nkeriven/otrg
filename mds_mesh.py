import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch

import utils.ot as uot
import utils.data as udata
import utils.plot as uplt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
plt.close('all')
savefig = True
np.random.seed(0)

##################################
filename = 'shrec__14_0'

G, pos = udata.load_mesh('data/'+filename+'.obj')

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(projection='3d')
uplt.my_draw3d(G, pos=pos, edge_color='k',width=.8,
              node_size=100, alpha_edge=.3, alpha=.5, ax=ax,
              node_color=[G.nodes[n]['label'] for n in G.nodes], cmap='jet')
if savefig:
    plt.savefig(f'fig/mesh_{filename}_seg.png', bbox_inches=0)

y = np.array([G.nodes[n]['label'] for n in G.nodes])
n = len(G)
Chat, SP = uot.SP_matrix(G, device=device, pos=pos)

n_comm = int(y.max())+1
dist = np.zeros((n_comm, n_comm, 2))
epsilon=.01
Gdist = [nx.empty_graph(n_comm), nx.empty_graph(n_comm)]
e_mult = [10,10]
sigma_dist = [.2,.2]
for i in range(n_comm):
    for j in range(i):
        indi = np.where(y==i)[0]
        indj = np.where(y==j)[0]
        C = Chat[np.ix_(indi, indj)]
        Phat, _, _, dist[i,j,0] = uot.sinkhorn_dual(C, epsilon=epsilon, device=device)
        Gdist[0].add_edge(i, j, weight=e_mult[0]*np.exp(-dist[i,j,0]**2/sigma_dist[0]))

        dist[i,j,1] = C.cpu().numpy().sum()/(len(indi)*len(indj))
        Gdist[1].add_edge(i, j, weight=e_mult[1]*np.exp(-dist[i,j,1]**2/sigma_dist[1]))


for k in range(2):
    pos2 = nx.spring_layout(Gdist[k], weight='weight')
    plt.figure(figsize=(10,10))
    options = {
        "font_size": 25,
        "node_size": 3000,
        "edgecolors": "black",
        "width": [Gdist[k].edges[i,j]['weight'] for (i,j) in Gdist[k].edges()],
        "pos" :pos2,
        "node_color":np.arange(n_comm),
        "cmap":'jet'
    }
    nx.draw(Gdist[k], **options)
    if savefig:
        plt.savefig(f'fig/mesh_{filename}_mds_{k}.png', bbox_inches=0)

####
i,j = 0,11
indi = np.where(y==i)[0]
indj = np.where(y==j)[0]
C = Chat[np.ix_(indi, indj)]
Phat, _, _, dist = uot.sinkhorn_dual(C**2, epsilon=.01, device=device)

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(projection='3d')
size = np.zeros(n)
size[indi]=100
size[indj]=100
uplt.my_draw3d(G, pos=pos, edge_color='k',width=.8*np.sqrt(200/n),
              node_size=size, alpha_edge=.3, alpha=.5, ax=ax,
              node_color=[G.nodes[n]['label'] for n in G.nodes])

# e_weights = np.zeros((n,n))
# for ii, _i in enumerate(indi):
#     for jj, _j in enumerate(indj):
#         p = SP[_i][_j]
#         subG = G.subgraph(p)
#         for e in subG.edges:
#             e_weights[e[0], e[1]] += 20*Phat[ii,jj].item()

# e_weights += e_weights.T
# ee_weights = [e_weights[e[0], e[1]] for e in G.edges]
e_weights = uplt.compute_edge_weights_SP(G, Phat, SP, n, indi, indj)

uplt.my_draw3d(G, pos=pos, edge_color='r', width=e_weights,
               node_size=0, alpha_edge=1, ax=ax)

if savefig:
    plt.savefig(f'fig/mesh_{filename}_illus.png', bbox_inches=0)


# plt.figure()
# nx.draw(nx.empty_graph(n_comm),
#         pos=np.zeros((n_comm,2))+np.arange(n_comm)[:,None],
#         node_color=np.arange(n_comm),
#         cmap='jet')