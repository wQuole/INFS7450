import time
import numpy as np
import networkx as nx
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DATASET = './data/facebook.txt'


def load_edges(filepath, type=int):
    """
    Load edges from a file (e.g. .txt), with a datatype=type
    Returns a undirected Graph with nodes and edges
    """
    G = nx.read_edgelist(path=filepath, nodetype=type)
    G = G.to_undirected()
    return G


def inv_dict(d):
    """
    Helper method to invert values in a dict
    """
    ret = {}
    for node, degree in d.items():
        ret[node] = 1/degree
    return ret


def get_top_n_nodes(nodes, n=10, nx=False):
    """
    Helper method to return top n nodes from a dict
    """
    top = OrderedDict(sorted(nodes.items(), key=lambda kv: kv[1], reverse=True)[:n])
    for node, pagerank in top.items():
        if nx:
            print(f"Node-{node} --> \t C_b: {pagerank}")
        else:
            print(f"Node-{node} --> \t C_b: {pagerank/len(nodes)}")
    return [n for n in top.keys()]


def page_rank(G, alpha=0.85, epsilon=1e-4, max_iter=100):
    """ PageRank on undirected graph with L2 norm, assuming no dangling nodes
    Original paper: http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf
    Helpful paper: http://home.ie.cuhk.edu.hk/~wkshum/papers/pagerank.pdf
    Args:
        G (networkx.classes.graph.Graph): Collection of nodes and edges
        alpha (float): dampening factor
        epsilon (float): threshold value for convergence
        max_iter (int): maximum allowed number of iterations
    Returns:
        A dict containing nodes and their respective PageRank
        {0: 24.680865117351715, ..., 4038: 1.1616657299654847}
    """
    N = G.number_of_nodes()
    A = G.adj
    D = dict(d for d in G.degree)
    D = inv_dict(D)
    c = dict((n, 1.0) for n in G)
    for _ in range(max_iter):
        prev_c = c
        c = dict((n, 0.0) for n in G)
        for node in A:
            for nbr in A[node]:
                c[nbr] += alpha * D[node] * prev_c[node]
            c[node] += (1 - alpha)
        delta = (np.linalg.norm([c[n] - prev_c[n] for n in c]))
        if delta < N * epsilon:
            return c
    print(f"Did not converge in {max_iter} iterations.")


def main():
    G = load_edges(DATASET)

    start = time.time()
    pr = page_rank(G)
    end = time.time()
    print(f"wQuole PageRank took {round(end - start, 3)} seconds to run.")
    top = get_top_n_nodes(pr)


    start = time.time()
    nx_pr = nx.pagerank(G)
    end = time.time()
    print(f"NetworkX PageRank took {round(end - start, 3)} seconds to run.")
    get_top_n_nodes(nx_pr, nx=True)

    # Write to file
    # with open("output/46301303.txt", "a") as txtfile:
    #     for node in top:
    #         print(node, end=" ", file=txtfile)


main()