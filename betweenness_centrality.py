import time
import networkx as nx
from collections import deque
from collections import OrderedDict

DATASET = './data/facebook.txt'

def load_edges(filepath):
    G = nx.read_edgelist(path=filepath, nodetype=int)
    vertices = G.nodes()
    neighbors = G.adj
    return G, vertices, neighbors



def brandes_algorithm(vertices, neighbors):
    '''
    Ulrik Brandes (2001) A faster algorithm for betweenness centrality , Journal of
    Mathematical Sociology, 25:2, 163-177, DOI: 10.1080/0022250X.2001.9990249
    https://doi.org/10.1080/0022250X.2001.9990249
    '''
    C_b = dict((v, 0) for v in vertices) # same as {v: 0 for v in vertices}
    for s in vertices:
        S = deque()  # Stack --> use pop()
        P = dict((w, []) for w in vertices)  # predecessors
        sigma = dict((t, 0) for t in vertices)  # number of shortest paths
        sigma[s] = 1
        delta = dict((t, -1) for t in vertices) #
        delta[s] = 0
        Q = deque([])  # Queue --> use popleft()
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

def normalize(n,value):
    return (1/((n-1)*(n-2)))*value

def main():
    G, vertices, neighbors = load_edges(DATASET)
    start = time.time()
    betweenness_centrality = brandes_algorithm(vertices, G)
    end= time.time()
    print(f"Exectued Brandes Algorithm in {end-start} seconds")
    top_10_nodes = OrderedDict(sorted(betweenness_centrality.items(), key=lambda kv: kv[1], reverse=True)[:10])
    N = len(G)
    for node, cb in top_10_nodes.items():
        print(f"Node-{node} -->\t\t C_b: {round(normalize(N, cb), 10)}")

    nx_cb = nx.betweenness_centrality(G)
    nx_top_10_nodes = OrderedDict(sorted(nx_cb.items(), key=lambda kv: kv[1], reverse=True)[:10])
    for node, cb in nx_top_10_nodes.items():
        print(f"Node-{node} -->\t\t C_b: {cb}")

main()
