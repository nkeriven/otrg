# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import torch

import pickle
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

ndist = 10 # size of distrib
nbary = 10 # size of barycenter support
epsilon = .025
weights = np.array([.1, .3, .2, .4])
colors = uplt.mycolor()

ds = [2,3,5]
ns = np.logspace(2,3.2,15, dtype=int) # number of additional points
nexp = 50

filename = 'bary_curve_res.pkl'

for _ in range(nexp):
    res = np.zeros((len(ds), len(ns), 1))
    theorieres = np.zeros((len(ds), len(ns), 1))
    for dind, d in enumerate(ds):
        XX = udata.normalize(np.random.randn(4*ndist+nbary,d))
    
        # distributions
        dist = dict()
        for s in range(4):
            # XX[ndist*s:ndist*(s+1), :] = udata.deform_data(np.array([(-1)**s*0.8, (-1)**(int(s/2))*0.6])[None,:]
            #                                                 + .2*(2*np.random.rand(ndist,2)-1))
            dist[s] = torch.rand(ndist, device=device, dtype=torch.float64) # weights
            dist[s] /= dist[s].sum()
            
        # true barycenters
        trueC = np.arccos(XX @ XX.T)
        np.fill_diagonal(trueC, 0)
        trueC = torch.tensor(trueC, device=device)**2
        trueCs = dict()
        for s in range(4):
            trueCs[s] = trueC[4*ndist:4*ndist+nbary,
                              ndist*s:ndist*(s+1)]**2
    
        # barycenter computation
        truebary, truePs = uot.barycenters(trueCs, dist, weights, device=device, epsilon=epsilon,
                                           n_iter=1000, same_space=False)
        
        for nind, n in enumerate(ns):
            print(dind, nind, _)
            # additional points
            YY = udata.normalize(np.random.randn(n,d))
            X = np.concatenate((XX,YY), axis=0)
        
            h = 1.3*n**(-1/(1.5*d))
            # graph
            G, h = udata.connected_eps_graph(X, h = h)
        
            # compute shortest path
            C = torch.zeros((n,n), device=device, dtype=torch.float64)
            print('Compute shortest paths...')
            # C, SP = uot.SP_matrix(G, device=device, h=h)
        
            Cs = dict()
            for s in range(4):
                Cs[s] = uot.SP_matrix(G, device=device, h=h, indices = (np.arange(4*ndist,4*ndist+nbary),
                                      np.arange(ndist*s,ndist*(s+1))))[0]**2
                # Cs[s] = C[4*ndist:4*ndist+nbary,
                #           ndist*s:ndist*(s+1)]**2
        
            # barycenter computation
            bary, Ps = uot.barycenters(Cs, dist, weights, device=device, epsilon=epsilon,
                                       n_iter=1000, same_space=False)
            
            res[dind, nind, 0] = np.linalg.norm(bary.detach().cpu().numpy()-truebary.detach().cpu().numpy())
            theorieres[dind, nind, 0] = (h + (n*h**d)**(-1/d))/5
    if _==0:
        r = res 
        tr = theorieres 
    else:
        r = np.concatenate((r, res), axis=2)
        tr = np.concatenate((tr, theorieres), axis=2)
    with open(filename, 'wb') as f:
        pickle.dump([ds, ns, r, tr], f)
     
with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
    ds, ns, rr, rrr = pickle.load(f)
rr = (rr**2).mean(axis=2)
rrr = rrr.mean(axis=2)
# rr = np.median(rr**2, axis=2)
# rrr = np.median(rrr, axis=2)
plt.figure()
cs=['b','r','g']
for (dind, d) in enumerate(ds):
    plt.loglog(ns,rr[dind,:], label=f'd={d}', c=cs[dind], linewidth=3)
    plt.loglog(ns,rrr[dind,:]/5,'--', label=f'd={d} (theory)', c=cs[dind], linewidth=3)
plt.legend(fontsize=12)
plt.xlabel('N', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.savefig('fig/bary_curve.pdf', bbox_inches='tight')


# # obj0, obj1, obj2 are created here...


# # Saving the objects:
# with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump([res, theorieres], f)

# with open(filename, 'rb') as f:  # Python 3: open(..., 'wb')
#     rr, rrr = pickle.load(f)

