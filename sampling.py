"""Functions to help with sampling trees."""

import pickle
import numpy as np
import random
import example
import numToNum as nTn

def gen_samples(trees, labels, vectors, vector_lookup):
    """Creates a generator that returns a tree in BFS order with each node
    replaced by its vector embedding, and a child lookup table."""

    # encode labels as one-hot vectors
    label_lookup = {label: _onehot(i, len(labels)) for i, label in enumerate(labels)}

    for tree in trees:
        nodes = []
        children = []
        label = label_lookup[tree['label']]

        queue = [(tree['tree'], -1)]
        while queue:
            node, parent_ind = queue.pop(0)
            node_ind = len(nodes)
            # add children and the parent index to the queue
            queue.extend([(child, node_ind) for child in node['children']])
            # create a list to store this node's children indices
            children.append([])
            # add this child to its parent's child list
            if parent_ind > -1:
                children[parent_ind].append(node_ind)
            nodes.append(vectors[vector_lookup[node['node']]])

        yield (nodes, children, label)

def gen_samplesint(vectors):
    length = example.intProg(0)
    prog_index = 1
    nodes = []
    children = []
    child = []
    first = 0
    while prog_index < length:
        temp = example.intProg(prog_index)
        # print("gen_samplesint\n")
        # print(temp)
        if temp != -1:
            if first == 0:
                temp = nTn.NODE_MAP_[temp]
                nodes.append(vectors[temp])
                first = 1
            else:
                if (temp != -2):
                    child.append(temp)
        else:
            children.append(child)
            child = []
            first = 0
        prog_index = prog_index + 1
    children[0] = children[0][1:]
    return nodes, children

def gen_samples1(ast, vectors, vector_lookup):
    """Creates a generator that returns a tree in BFS order with each node
    replaced by its vector embedding, and a child lookup table."""

    # encode labels as one-hot vectors
    # label_lookup = {label: _onehot(i, len(ast)) for i, label in enumerate(ast)}
    nodes = []
    children = []
    for tree in ast:
        # label = label_lookup[tree['label']]
        node = tree['name']
        nodes.append(vectors[vector_lookup[node]])
        children.append(tree['children'])
    return nodes, children

def batch_samples(gen, batch_size):
    """Batch samples from a generator"""
    nodes, children, labels = [], [], []
    samples = 0
    for n, c, l in gen:
        nodes.append(n)
        children.append(c)
        labels.append(l)
        samples += 1
        if samples >= batch_size:
            yield _pad_batch(nodes, children, labels)
            nodes, children, labels = [], [], []
            samples = 0

    if nodes:
        yield _pad_batch(nodes, children, labels)

def _pad_batch(nodes, children):
    if not nodes:
        return [], [], []
    max_nodes = max([len(x) for x in nodes])
    max_children = max([len(x) for x in children])
    feature_len = len(nodes[0][0])
    child_len = max([len(c) for n in children for c in n])

    nodes = [n + [[0] * feature_len] * (max_nodes - len(n)) for n in nodes]
    # pad batches so that every batch has the same number of nodes
    children = [n + ([[]] * (max_children - len(n))) for n in children]
    # pad every child sample so every node has the same number of children
    children = [[c + [0] * (child_len - len(c)) for c in sample] for sample in children]

    return nodes, children

def _onehot(i, total):
    return [1.0 if j == i else 0.0 for j in range(total)]