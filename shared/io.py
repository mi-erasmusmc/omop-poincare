#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared core for input-output.
"""
# --------------------------------------------------------------------------- #
#                  MODULE HISTORY                                             #
# --------------------------------------------------------------------------- #
# Version          1
# Date             2021-08-05
# Author           LH John
# Note             Original version
#
# --------------------------------------------------------------------------- #
#                  SYSTEM IMPORTS                                             #
# --------------------------------------------------------------------------- #
import pathlib

# --------------------------------------------------------------------------- #
#                  OTHER IMPORTS                                              #
# --------------------------------------------------------------------------- #
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
from tqdm import tqdm
from deprecated import deprecated

# --------------------------------------------------------------------------- #
#                  OWN IMPORTS                                                #
# --------------------------------------------------------------------------- #
# from configs.config import extern

# --------------------------------------------------------------------------- #
#                  META DATA                                                  #
# --------------------------------------------------------------------------- #
__status__ = 'Development'

# --------------------------------------------------------------------------- #
#                  CONSTANTS                                                  #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#                  EXPORTED FUNCTIONS                                         #
# --------------------------------------------------------------------------- #
def read_ref(path):
    """Load references of edge list data.
    :param path: Path to reference of the form [concept_id, concept_name]
    :return:
    """
    return pd.read_csv(path)


@deprecated
def load_edge_list(path, symmetrize=False):
    """Read edge list data from comma-separated file.
    :param path: Path to the edge list of form [id1, id2, weight], where id1
    are the descendant, id2 are the ancestor, and weight are the graph
    distances.
    :param symmetrize:
    :return:
        - idx - factorized edge list of data
        - objects - list of unique concept identifiers
        - weights - weights of concepts
    """
    df = pd.read_csv(path, usecols=['id1', 'id2', 'weight'], engine='c')
    df.dropna(inplace=True)
    if symmetrize:
        rev = df.copy().rename(columns={'id1' : 'id2', 'id2' : 'id1'})
        df = pd.concat([df, rev])
    idx, objects = pd.factorize(df[['id1', 'id2']].values.reshape(-1))
    idx = idx.reshape(-1, 2).astype('int')
    weights = df.weight.values.astype('float')
    return idx, objects.tolist(), weights


def read_data(path, skip_stats=True, remove_duplicates=False):
    """Read edge list data from comma-separated file.
    :param path: Path to the edge list of form [id1, id2, weight], where id1
    are the descendant, id2 are the ancestor, and weight are the graph
    distances.
    :param skip_stats: Skip showing some additional statistics of the data.
    :param remove_duplicates: Remove duplicate data.
    :return:
        - ids - factorized edge list of data
        - weights - weights of concepts
        - objects - list of unique concept identifiers
        - neighbors - list of neighbors per concept
        - diff_summed -
        - relations -
    """
    print("Processing dataset...")

    if isinstance(path, pathlib.PurePath):
        path = str(path.resolve())

    if "hyperlex" in path:
        df = pd.read_csv(path, usecols=[0, 1, 3], engine='c',
                         delim_whitespace=True)
    else:
        df = pd.read_csv(path, usecols=[0, 1], engine='c', escapechar='\\',
                         na_filter=False)

    ids, names = pd.factorize(df.iloc[:, :2].values.reshape(-1))
    ids = ids.reshape(-1, 2).astype('int')

    # three possibilities:
    # 1. how many times does w_1 occur
    # 2. how many times does w_2 occur
    # 3. how many times does (w_1, w_2) occur
    # 4. how many times does w_1 + w_2 occur <-- here
    # word2vec uses occurrences of w_2 (context)
    weights = np.bincount(ids.flatten()).astype(np.float32)
    weights /= np.sum(weights)

    if "hyperlex" in path:
        ids = ids[(df.iloc[:, 2] != "no-rel") & (df.iloc[:, 2] != "ant")]

    voc_size = len(names)
    adjacency_matrix = lil_matrix((voc_size, voc_size), dtype=np.uint8)
    adjacency_loop = lil_matrix((voc_size, voc_size), dtype=np.uint8)
    for i, row in enumerate(ids):
        adjacency_matrix[row[0], row[1]] = 1

        adjacency_loop[row[0], row[1]] = 1
        adjacency_loop[row[0], row[0]] = 1
        adjacency_loop[row[1], row[1]] = 1

    neighbors = []
    for i in range(voc_size):
        neighbors.append(adjacency_matrix[i].nonzero()[1])

    r = np.array(adjacency_matrix.sum(axis=1).astype(np.float32))
    diff_summed = (r * (r - 1) / 2).sum()

    relations = adjacency_matrix.sum()

    print(f"Edges: {df.shape[0]}")
    print(f"Relations: {relations}")
    print(f"Nodes: {voc_size}")
    # if relations < 10000:
    #     compute_hyperbolicity_and_other_stats(adjacency_matrix)

    if remove_duplicates:
        ids = np.array(list(set([tuple(x) for x in ids.tolist()])))

    # print(f"Hyperbolicity sample: {compute_avg_hyperbolicity_sample(adjacency_matrix)}")

    if skip_stats:
        return ids, weights, names.tolist(), neighbors, diff_summed, relations

    print("Top 10 number of longest hypernyms...")
    path_lens = []
    for q in tqdm(range(voc_size)):
        out = []
        reachable = adjacency_loop[q].nonzero()[1].tolist()
        while reachable:
            node = reachable.pop()
            if node in out:
                continue
            out.append(node)
            reachable.extend(adjacency_loop[node].nonzero()[1].tolist())
        path_lens.append((len(out), names[q]))

    path_lens.sort(reverse=True)
    print(path_lens[:10])

    print("Top 10 number of shortest hypernyms...")
    print(path_lens[-10:])

    return ids, weights, names.tolist(), neighbors, diff_summed, relations


def get_dict_data(objects, ref, dict_type="name"):
    """Create a dictionary of concept identifiers and concept labels.
    :param objects: List of unique concept identifiers.
    :param ref: List of unique concept labels.
    :param dict_type: "name" for string names or "concept" for concept IDs
    :return: Dictionary of concept identifiers and concept labels.
    """
    dict_data = dict(enumerate(objects))
    print(f"dictData[1]: {dict_data[1]}")

    if dict_type == "name":
        # replace concept_id with concept_name in dict
        for key, value in dict_data.items():
            dict_data[key] = ref.loc[ref['concept_id'] == value].concept_name.values[0]

    return dict_data
