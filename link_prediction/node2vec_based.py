# Local modules
import utils
# External modules
import numpy as np
import networkx as nx
from random import choice
from node2vec import Node2Vec
from collections import OrderedDict
from node2vec.edges import HadamardEmbedder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


TRAINING_POS = "data/training.txt"
TRAINING_NEG = "data/training_negative.txt"
VAL_POS = "data/val_positive.txt"
VAL_NEG = "data/val_negative.txt"
TESTING = "data/testing.txt"


def get_graph(filepath=None, lists=None):
    if lists is not None:
        nx_graph = nx.Graph()
        nx_graph.add_edges_from(lists)
    else:
        nx_graph = nx.read_edgelist(path=filepath, nodetype=int, delimiter=' ')
        nx_graph = nx_graph.to_undirected()
    return nx_graph


def get_neighbors(G):
    neighbors = dict()
    for node in G:
        neighbors[node] = list(nx.neighbors(G, node))
    return neighbors


def get_edges(filepath, label=None):
    x, y = [], []
    with open(filepath, 'r') as pairs:
        for pair in pairs:
            n1, n2 = pair.strip().split()
            if label is not None:
                x.append((int(n1), int(n2)))
                y.append(label)
            else:
                x.append((int(n1), int(n2)))
    if label is not None:
        return x, y
    else:
        return x


def merge_datasets(pos_path, neg_path, labelling=False):
    if labelling:
        x_pos, y_pos = get_edges(pos_path, label=1)
        x_neg, y_neg = get_edges(neg_path, label=0)
        x = x_pos + x_neg
        y = y_pos + y_neg
        return x, y
    else:
        pos = get_edges(pos_path)
        neg = get_edges(neg_path)
        x = pos + neg
        return x


def generate_negative_edges(G_train, complete_data, N):
    new_negative_edges = []

    while len(new_negative_edges) < 5*N:
        node1 = choice(list(G_train.nodes()))
        node2 = choice(list(G_train.nodes()))
        new_link = (node1, node2)
        if new_link in complete_data or new_link[::-1] in complete_data:
            print(f"{new_link} already in dataset.")
        else:
            new_negative_edges.append(new_link)
    return new_negative_edges


def get_top_n_links(links, n=100, out=False):
    top_links = OrderedDict(sorted(links.items(), key=lambda kv: kv[1], reverse=True)[:n])
    if out:
        for link, link_score in top_links.items():
            print(f"{link} --> {link_score}")
    return [n for n in top_links.keys()]


def evaluate(proposed, ground_truth):
    hits = 0
    for i, link in enumerate(proposed):
        if link in ground_truth:
            print(f"{link} <-- {link[::-1]}")
            hits += 1
    print(f"{hits}/{len(ground_truth)}")
    return float(hits/len(ground_truth))


def preprocess_lists(X_train, cast=str):
    ret = []
    for pair in X_train:
        a1, a2 = cast(pair[0]), cast(pair[1])
        ret.append((a1, a2))
    return ret


if __name__ == '__main__':
    '''PREPARE DATA'''
    # --- Training data ---
    # Generate graph and lists for positive training samples
    G_train_pos = get_graph(filepath=TRAINING_POS)
    X_train_pos = list(G_train_pos.edges())

    # Generate graph and lists for negative + positive training samples
    X_train, y_train = merge_datasets(TRAINING_POS, TRAINING_NEG, labelling=True)
    G_train = get_graph(lists=X_train)
    nbrs_train = get_neighbors(G_train)

    # --- Validation ---
    X_validation = merge_datasets(VAL_POS, VAL_NEG, labelling=False)
    G_validation = get_graph(lists=X_validation)
    links_validation = list(G_validation.edges())
    # For evaluation matters later
    G_validation_pos = get_graph(filepath=VAL_POS)
    positive_validation_links = list(G_validation_pos.edges())
    G_validation_neg = get_graph(filepath=VAL_NEG)
    negative_validation_links = list(G_validation_neg.edges())


    '''node2vec'''
    n2v = Node2Vec(G_train,
                   dimensions=64,
                   walk_length=16,
                   num_walks=10,
                   p=1,
                   q=1)
    print("Fitting model...")
    model = n2v.fit(window=5, min_count=1, sg=1, hs=0)
    print("Embedding nodes...")
    hadamard_embedded_links = HadamardEmbedder(keyed_vectors=model.wv)
    links_validation = preprocess_lists(links_validation)
    links_validation_embedded = [hadamard_embedded_links[x] for x in links_validation]
    X_train = preprocess_lists(X_train)
    X_train_embedded = [hadamard_embedded_links[x] for x in X_train]

    '''CLASSIFICATION'''
    # Standardize features by removing mean and scale to unit variance
    std = StandardScaler()
    X_train = std.fit_transform(X_train_embedded)
    X_validation = std.transform(links_validation_embedded)

    logit = LogisticRegression(solver='saga', max_iter=500)
    clf = logit.fit(X_train, y_train)
    probabilities = clf.predict_proba(X_validation)

    '''EVALUATION'''
    score = dict()
    links_validation = preprocess_lists(links_validation, cast=int)
    for i, prob in enumerate(probabilities):
        score[links_validation[i]] = prob[1]
    top100 = get_top_n_links(score, out=1)

    # Use positive validation set to evaluate score
    G_valid_pos = get_graph(VAL_POS)
    links_ground_truth = list(G_valid_pos.edges())

    # Evalute accuracy
    accuracy = evaluate(top100, links_ground_truth)
    print(f"Accuracy on validation set: {round(accuracy*100, 5)} %.")


    '''GENERATE TRAINING DATA'''
    # dont_touch = links_validation + links_testing + X_train_pos
    # new_training_data = generate_negative_edges(G_train, dont_touch, N=len(X_train_pos))
    # with open('data/training_negative.txt', 'w') as out:
    #     for pair in new_training_data:
    #         print(str(pair[0]) + ' ' + str(pair[1]), end='\n', file=out)
