#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 11:45:07 2022

@author: kerivenn
"""

import numpy as np
import networkx as nx
import torch

from numba import jit
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from scipy.integrate import quad

#%% random graphs

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

#%% generate latent positions

def pdf(x, symmetric=False):
    return (x**2+.2)/1.2

class my_distribution(stats.rv_continuous):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # integrate area of the PDF in range a..b
        self.scale, _ = quad(lambda x: pdf(x), self.a, self.b)

    def _pdf(self, x):
        # scale PDF so that it integrates to 1 in range a..b
        return pdf(x) / self.scale

distribution = my_distribution(a=-1, b=1)

def deform_data(X):
    Y = X.copy()
    Y[:,1] *= (Y[:,0]**2+.2)/1.2
    return Y

def tube_data(n):
    X = 2*np.random.rand(n,2)-1
    X[:,0] = distribution.rvs(size=n)
    return deform_data(X)

def generate_two_circles(n, noise=.1, noise_out=0):
    X = np.random.randn(3*n,2)
    # inner circle
    X[:n,:] = .6*X[:n,:]*((1+noise*np.random.randn(n))/(np.sqrt((X[:n,:]**2).sum(axis=1))))[:,None]
    # outer circle
    X[n:,:] = X[n:,:]*((1+noise_out*np.random.randn(2*n))/(np.sqrt((X[n:,:]**2).sum(axis=1))))[:,None]
    return X

def GMM(n, shift=1):
    X = np.zeros((3*n,2))
    X[:n,:] = np.random.randn(n,2)+np.array([shift,0])[None,:]
    X[n:,:] = 1.5*np.random.randn(2*n,2)+np.array([-shift,0])[None,:]
    return X

def normalize(X):
    X = X/np.sqrt((X**2).sum(axis=1))[:,None]
    return X

def generate_sphere(n):
    """Non-uniform distrib on the sphere"""
    X = normalize(np.random.randn(n,3)+np.array([.3,0,0])[None,:])
    return X

#%% mesh utils


def parse_obje(obj_file, scale_by=0):
    vs = []
    faces = []
    edges = []
    V = np.array

    def add_to_edges():
        if edge_c >= len(edges):
            for _ in range(len(edges), edge_c + 1):
                edges.append([])
        edges[edge_c].append(edge_v)

    def fix_vertices():
        nonlocal vs, scale_by
        vs = V(vs)
        z = vs[:, 2].copy()
        vs[:, 2] = vs[:, 1]
        vs[:, 1] = z
        max_range = 0
        for i in range(3):
            min_value = np.min(vs[:, i])
            max_value = np.max(vs[:, i])
            max_range = max(max_range, max_value - min_value)
            vs[:, i] -= min_value
        if not scale_by:
            scale_by = max_range
        vs /= scale_by

    with open(obj_file) as f:
        for line in f:
            line = line.strip()
            splitted_line = line.split()
            if not splitted_line:
                continue
            elif splitted_line[0] == 'v':
                vs.append([float(v) for v in splitted_line[1:]])
            elif splitted_line[0] == 'f':
                faces.append([int(c) - 1 for c in splitted_line[1:]])
            elif splitted_line[0] == 'e':
                if len(splitted_line) >= 4:
                    edge_v = [int(c) - 1 for c in splitted_line[1:-1]]
                    edge_c = int(splitted_line[-1])
                    add_to_edges()

    vs = V(vs)
    fix_vertices()
    faces = V(faces, dtype=int)
    edges = [V(c, dtype=int) for c in edges]
    return (vs, faces, edges), scale_by

def mesh2graph(mesh):
    vs, _, edges = mesh
    n = len(vs)
    G = nx.empty_graph(n)
    for edge_c, edge_group in enumerate(edges):
        for edge in edge_group:
            G.add_edge(edge[0], edge[1], label=edge_c)
    return G

def relabel_nodes(G):
    # from edge label to node label
    max_label=0
    for e in G.edges:
        G.nodes[e[0]]['label'] = G.edges[e]['label']
        G.nodes[e[1]]['label'] = G.edges[e]['label']
        if G.edges[e]['label'] > max_label:
            max_label = G.edges[e]['label']

    sup_label = max_label + 1
    for l in range(max_label+1):
        SG = nx.subgraph(G, [n for n in G.nodes if G.nodes[n]['label']==l])
        SGs = [G.subgraph(c).copy() for c in nx.connected_components(SG)]
        if len(SGs)>1:
            for i in range(1,len(SGs)):
                for n in SGs[i].nodes:
                    G.nodes[n]['label'] = sup_label
                sup_label += 1

    return G

def check_well_clustered(G):
    set_label = set([G.nodes[n]['label'] for n in G.nodes])
    for l in set_label:
        SG = nx.subgraph(G, [n for n in G.nodes if G.nodes[n]['label']==l])
        if not nx.is_connected(SG):
            return False
    return True

def load_mesh(filename):
    mesh, _ = parse_obje(filename)
    G = mesh2graph(mesh)
    G = relabel_nodes(G)
    return G, mesh[0]

