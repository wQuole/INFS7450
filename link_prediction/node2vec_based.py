import numpy as np
import pandas as pd
import networkx as nx
from random import choice
from node2vec import Node2Vec
from collections import OrderedDict
from node2vec.edges import HadamardEmbedder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


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


def preprocess_for_hadamard(X_train):
    ret = []
    for pair in X_train:
        a1, a2 = str(pair[0]), str(pair[1])
        ret.append((a1, a2))
    return ret


def node2vec_embedding(G, validation_links, dims=20, how_long=20, number_of_walks=20, return_parameter=1.4, walk_away_parameter=0.9):
    n2v = Node2Vec(G, dimensions=dims, walk_length=how_long, num_walks=number_of_walks, p=return_parameter, q=walk_away_parameter)
    print("Fitting model...")
    model = n2v.fit(window=4, min_count=1, sg=1, hs=0)
    return model


if __name__ == '__main__':
    '''TRAINING'''
    # Generate graph and lists for positive training samples
    G_train_pos = get_graph(filepath=TRAINING_POS)
    X_train_pos = list(G_train_pos.edges())

    # Generate graph and lists for negative + positive training samples
    X_train, y_train = merge_datasets(TRAINING_POS, TRAINING_NEG, labelling=True)
    G_train = get_graph(lists=X_train)
    nbrs_train = get_neighbors(G_train)

    '''VALIDATION'''
    X_validation = merge_datasets(VAL_POS, VAL_NEG, labelling=False)
    G_validation = get_graph(lists=X_validation)
    links_validation = list(G_validation.edges())
    # For evaluation matters later
    G_validation_pos = get_graph(filepath=VAL_POS)
    positive_validation_links = list(G_validation_pos.edges())
    G_validation_neg = get_graph(filepath=VAL_NEG)
    negative_validation_links = list(G_validation_neg.edges())

    '''node2vec'''
    model = node2vec_embedding(G_train, links_validation)
    edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
    X_train = preprocess_for_hadamard(X_train)
    X_train_embedded = [edges_embs[(x)] for x in X_train]


    '''EVALUATION'''
    links_validation = preprocess_for_hadamard(links_validation)
    links_validation_embedded = [edges_embs[(x)] for x in links_validation]
    lr = LogisticRegression(class_weight='balanced')
    clf = lr.fit(X_train_embedded, y_train)

    predictions = clf.predict_proba(links_validation_embedded)
    print("predictions\n",predictions)
    # Use positive validation set to evaluate score
    G_valid_pos = get_graph(VAL_POS)
    links_ground_truth = list(G_valid_pos.edges())
    validation_edge_labels = np.concatenate([np.ones(len(positive_validation_links)), np.zeros(len(negative_validation_links))])
    print(clf.score(links_validation_embedded, validation_edge_labels))
    # Evalute accuracy
    #top_validation = get_top_n_links(predictions, out=False)
    #eval_validation = evaluate(top_validation, links_ground_truth)
    #print(f"Accuracy: {round(eval_validation*100, 2)} % on the validation set.")

    '''TESTING'''
    # G_test = get_graph(TESTING)
    # links_testing = list(G_test.edges())
    # score_testing = compute_proximity(G_test, edges=links_testing, neighbors=nbrs_train, measurement='jaccard')
    #
    # top_testing = get_top_n_links(score_testing, out=False)
    # # Write to file
    # with open("output/node2vec/46301303.txt", "w") as txtfile:
    #      for pair in top_testing:
    #          print(str(pair[0])+' '+str(pair[1]), end="\n", file=txtfile)

    # '''GENERATE TRAINING DATA'''
    # dont_touch = links_validation + links_testing + X_train_pos
    # new_training_data = generate_negative_edges(G_train, dont_touch, N=len(X_train_pos))
    # with open('data/training_negative.txt', 'w') as out:
    #     for pair in new_training_data:
    #         print(str(pair[0]) + ' ' + str(pair[1]), end='\n', file=out)
