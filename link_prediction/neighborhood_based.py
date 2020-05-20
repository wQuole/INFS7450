import numpy as np
import pandas as pd
import networkx as nx
from collections import OrderedDict
from networkx.classes.function import neighbors as nx_find_neighbors

TRAINING = "data/training.txt"
VAL_POS = "data/val_positive.txt"
VAL_NEG = "data/val_negative.txt"
TESTING = "data/testing.txt"


def load_data(filepath, col_names=['source', 'target']):
    df = pd.read_csv(filepath, sep=" ", header=None, names=col_names, dtype=np.int64)
    nx_graph = nx.read_edgelist(path=filepath, nodetype=int)
    nx_graph = nx_graph.to_undirected()
    return df, nx_graph


def get_edges_for_validation(filepath):
    temp = []
    with open(filepath) as validation:
            for pair in validation:
                n1, n2 = pair.strip().split()
                temp.append((int(n1), int(n2)))
    return temp


def merge_validation_sets(pos_path, neg_path):
    pos = get_edges_for_validation(pos_path)
    neg = get_edges_for_validation(neg_path)
    validation_data = pos + neg

    return validation_data


def get_neighbors(G):
    neighbors = dict()
    for node in G:
        neighbors[node] = list(nx_find_neighbors(G, node))
    return neighbors


def compute_proximity(G, edges, neighbors, measurement='jaccard'):
    score = dict()

    if measurement.lower() == 'jaccard':
        for pair in edges:
            p1, p2 = pair
            numerator = len(set(neighbors[p1]).intersection(set(neighbors[p2])))
            denominator = len(set(neighbors[p1]).union(set(neighbors[p2])))
            score[(p1, p2)] = numerator/denominator

    elif measurement.lower() == 'adamic' or measurement.lower() == 'adar':
        adamic_adar_index = nx.adamic_adar_index(G, edges)
        for p1, p2, aai in adamic_adar_index:
            score[(p1, p2)] = aai

    elif measurement.lower() == 'preferential':
        for pair in edges:
            p1, p2 = pair
            score[(p1, p2)] = len(neighbors[p1]) * len(neighbors[p2])
    else:
        return f"Measurement: '{measurement}' is not valid. Use one of the following:" \
               f"\n- jaccard" \
               f"\n- adamic" \
               f"\n- adar" \
               f"\n- preferential"
    return score


def get_top_n_links(links, n=100, out=False):
    top_links = OrderedDict(sorted(links.items(), key=lambda kv: kv[1], reverse=True)[:n])
    if out:
        for link, link_score in top_links.items():
            print(f"{link} --> {link_score}")
    return [n for n in top_links.keys()]


def evaluate(proposed, ground_truth):
    hits = 0
    for link in proposed:
        if link in ground_truth:
            hits += 1
    return hits/len(proposed)


if __name__ == '__main__':
    # TRAINING
    _, G_train = load_data(TRAINING)
    links_train = list(G_train.edges())
    nbrs_train = get_neighbors(G_train)

    # VALIDATION
    _, G_validation = load_data(VAL_POS)
    links_validation = list(G_validation.edges())
    score_validation = compute_proximity(G_train, edges=links_validation, neighbors=nbrs_train, measurement='adamic')

    top = get_top_n_links(score_validation)
    eval = evaluate(top, links_validation)

    print(f"Accuracy: {eval*100} % on the validation set.")

    # TESTING
    _, G_test = load_data(TESTING)
    links_testing = list(G_test.edges())
    score_testing = compute_proximity(G_test, edges=links_testing, neighbors=nbrs_train, measurement='adamic')

    top_test = get_top_n_links(score_testing, out=True)

    # Write to file
    with open("output/46301303.txt", "w") as txtfile:
         for pair in top_test:
             print(str(pair[0])+' '+str(pair[1]), end="\n", file=txtfile)
