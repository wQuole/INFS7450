import time
import networkx as nx
from collections import deque
from collections import OrderedDict
import matplotlib.pyplot as plt

DATASET = './data/facebook.txt'


def load_edges(filepath):
    """
    Load edges from a file (e.g. .txt), with a datatype=type
    Returns a undirected Graph with nodes and edges
    """
    G = nx.read_edgelist(path=filepath, nodetype=int)
    G = nx.to_undirected(G)
    vertices = G.nodes()
    neighbors = G.adj
    return G, vertices, neighbors


def normalize(n, value):
    """
    Normalize the Betweenness Centrality value
    """
    return value/((n - 1) * (n - 2))


def get_top_n_nodes(nodes, n=10, nx=False):
    """
    Helper method to return top n nodes from a dict
    """
    top = OrderedDict(sorted(nodes.items(), key=lambda kv: kv[1], reverse=True)[:n])
    for node, cb in top.items():
        if nx:
            print(f"Node-{node} --> \t C_b: {cb}")
        else:
            print(f"Node-{node} --> \t C_b: {round(normalize(len(nodes), cb), 10)}")
    return [n for n in top.keys()]


def brandes_algorithm(vertices, neighbors):
    """
    Ulrik Brandes (2001) A faster algorithm for betweenness centrality , Journal of
    Mathematical Sociology, 25:2, 163-177, DOI: 10.1080/0022250X.2001.9990249
    https://doi.org/10.1080/0022250X.2001.9990249
    """
    C_b = dict((v, 0) for v in vertices) # same as {v: 0 for v in vertices}
    for s in vertices:
        S = deque()  # Stack --> use pop()
        P = dict((w, []) for w in vertices)  # predecessors
        sigma = dict((t, 0) for t in vertices)  # number of shortest paths
        sigma[s] = 1
        delta = dict((t, -1) for t in vertices)
        delta[s] = 0
        Q = deque()  # Queue --> use popleft()
        Q.append(s)
        while Q:
            v = Q.popleft()
            S.append(v)
            for w in neighbors[v]:
                if delta[w] < 0:
                    Q.append(w)
                    delta[w] = delta[v] + 1
                if delta[w] == delta[v] + 1:
                    sigma[w] += sigma[v]
                    P[w].append(v)

        # S returns vertices in order of non-increasing distance from s
        dependency = dict((v, 0) for v in vertices)
        while S:
            w = S.pop()
            for v in P[w]:
                dependency[v] += (sigma[v]/sigma[w]) * (1 + dependency[w])
            if w != s:
                C_b[w] += dependency[w]
    return C_b


def main():
    G, vertices, neighbors = load_edges(DATASET)

    start = time.time()
    betweenness_centrality = brandes_algorithm(vertices, neighbors)
    end= time.time()
    print(f"wQuole Executed Brandes Algorithm in {end-start} seconds")
    top_10_nodes = get_top_n_nodes(betweenness_centrality)

    nx.draw_networkx(G, nodelist=top_10_nodes, with_labels=False)
    plt.savefig("cb.pdf")

    start = time.time()
    nx_cb = nx.betweenness_centrality(G)
    end= time.time()
    print(f"\nNetworkX Executed Brandes Algorithm in {end-start} seconds")
    get_top_n_nodes(nx_cb, nx=True)

    # Write to file
    # with open("output/46301303.txt", "a") as txtfile:
    #     for node in top_10_nodes:
    #         print(node, end=" ", file=txtfile)


main()
