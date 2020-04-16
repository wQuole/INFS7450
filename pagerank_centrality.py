import time
import networkx as nx
import numpy as np
from collections import OrderedDict


DATASET = './data/facebook.txt'

def load_edges(filepath):
    G = nx.read_edgelist(path=filepath, nodetype=int)
    G = G.to_undirected()

    A = nx.adj_matrix(G)
    A = A.todense()
    A = np.array(A, dtype=np.float64)

    D = np.diag(np.sum(A, axis=0))

    T = np.dot(np.linalg.inv(D), A)

    return G, A, D, T

def create_translation_matrix(G):
    A = nx.adj_matrix(G)
    A = A.todense()
    A = np.array(A, dtype=np.float64)

    D = np.diag(np.sum(A, axis=0))

    T = np.dot(np.linalg.inv(D), A)
    return T


def page_rank_1(G, alpha=0.85, beta=0.15, epsilon=1e-3):
    N = G.number_of_nodes()

    A = nx.adj_matrix(G)
    A = A.todense()
    A = np.array(A, dtype=np.float64)

    D = dict(G.degree())
    adj = dict(G.adj)

    c = dict((n, 1.0/N) for n in G)
    for _ in range(100):
        prev_c = c
        c = dict((n, 0.0) for n in G)
        for node in c:
            for nbr in adj[node]:
                #print("Nbr",nbr)
                #print(A[node][nbr])
                c[nbr] += (alpha * prev_c[nbr]/len(adj[node]))
            c[node] += beta
            delta = np.abs([np.sqrt(c[n]**2 + prev_c[node]**2) for n in c])
            #delta = np.abs(c[node] - prev_c[node])
            print(delta)
            if delta < epsilon:
                return c
    else:
        print("Did not converge.")


def page_ranko(G, alpha=0.85, beta=0.15, epsilon=1e-2):
    N = G.number_of_nodes()
    pagerank = dict((v, 1/N) for v in G)
    adj = G.adj
    while True:
        for n in adj:
            rank = 0
            #print("node",n)
            for v in adj[n]:
            #print("nbr",v)
                rank += alpha * pagerank[v]/len(adj[v])
            rank += beta
            diff = np.linalg.norm(pagerank[n] - rank)
            pagerank[n] = rank
        if diff < epsilon:
            break
    return pagerank


def page_rank(G, alpha=0.85, beta=0.15, epsilon=1e-4, max_iter=100):
    N = G.number_of_nodes()
    A = G.adj
    D = dict(d for d in G.degree())
    D = inv_dict(D)
    c = dict((n, 1.0/N) for n in G)
    for _ in range(max_iter):
        prev_c = c
        c = dict((n, 0.0) for n in G)
        for node in A:
            for nbr in A[node]:
                c[nbr] += alpha * prev_c[node] * D[node]
            c[node] += beta
        #delta = np.sum([np.power(c[n] - prev_c[n], 2) for n in c])
        delta = (np.linalg.norm([c[n] - prev_c[n] for n in c]))  # L2 Norm
        print(delta)
        if delta < N * epsilon:
            return c


def inv_dict(d):
    ret = {}
    for node, degree in d.items():
        ret[node] = 1/degree
    return ret


def get_top_n_nodes(nodes, n=10, nx=False):
    top = OrderedDict(sorted(nodes.items(), key=lambda kv: kv[1], reverse=True)[:n])
    for node, pagerank in top.items():
        if nx:
            print(f"Node-{node} --> \t C_b: {pagerank}")
        else:
            print(f"Node-{node} --> \t C_b: {pagerank/len(nodes)}")
def main():
    G, A, D, T = load_edges(DATASET)
    pr = page_rank(G)
    get_top_n_nodes(pr)

    print("\nNx:")
    nx_pr = nx.pagerank(G)
    get_top_n_nodes(nx_pr, nx=True)

    # deg = dict(G.degree())  # --> [(node, degree), ... (node, degree)]
    # for n, nbrsdict in G.adjacency():
    #     #print(n, "->", nbrsdict)
    #     print(nbrsdict)

    # for n in G.adj:
    #      print("Node?",n,"\n")
    #      for v in G.adj[n]:
    #         print("nbr?",v)
    #         print("OKEY???",G.adj[n][v])

    # a = [(n, nbrdict) for n, nbrdict in G.adjacency()]
    # for i in a:
    #     print(i[0],"-->",i[1])

    # for node in G.adj:
    #     print("NODE",node)
    #     for nbr in G.adj[node]:
    #         print("NBR", nbr)

main()