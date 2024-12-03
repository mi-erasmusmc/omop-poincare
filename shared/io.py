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
from scipy.sparse import coo_array
from tqdm import tqdm
from deprecated import deprecated
import torch
import os
import polars as pl

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
    return pd.read_csv(path, engine='pyarrow')


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
        df = pd.read_csv(path, engine='pyarrow', na_filter=False)

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
    adjacency_data = np.ones(ids.shape[0], dtype=np.bool_)
    adjacency_coords = (ids[:, 0], ids[:, 1])
    adjacency_matrix = coo_array((adjacency_data, adjacency_coords), (voc_size, voc_size))
    adjacency_matrix = adjacency_matrix.tocsr()

    adjacency_loop_data = np.ones(ids.shape[0] * 3, dtype=np.bool_)
    adjacency_loop_coords = (np.concatenate((ids[:, 0], ids[:, 0], ids[:, 1])),
                             np.concatenate((ids[:, 1], ids[:, 0], ids[:, 1])))
    adjacency_loop = coo_array((adjacency_loop_data, adjacency_loop_coords), (voc_size, voc_size))


    neighbors = [adjacency_matrix.indices[adjacency_matrix.indptr[i]:adjacency_matrix.indptr[i + 1]] for i in range(voc_size)]
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

def write_tensorflow_projector_data(model_path, ref_csv_path, output_dir='output'):
    state = torch.load(model_path)
    state_dict = state['state_dict']

    # Print all keys in the state dictionary to find the correct one
    print("Available keys in the state dictionary:")
    for key in state_dict.keys():
        print(key)

    embeddings = state_dict['embedding.weight'].cpu().numpy()
    names = state['concept_ids']

    print("First 100 embedding weights:")
    for i in range(100):
        print(f"Embedding {i}: {embeddings[i]}")

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'vecs.tsv'), 'w') as f_out:
        for record in embeddings:
            f_out.write('\t'.join(map(str, record)) + '\n')

    with open(os.path.join(output_dir, 'labels.tsv'), 'w') as f_out:
        for name in names:
            f_out.write(str(name) + '\n')

    ########
    # Polars operations for joining and getting the names
    ref_df = pl.read_csv(ref_csv_path)
    labels_df = pl.read_csv(os.path.join(output_dir, 'labels.tsv'), has_header=False, new_columns=['concept_id'])

    # Convert concept_id column to integer
    labels_df = labels_df.with_columns(pl.col('concept_id').cast(pl.Int64))

    # Perform the join to get the names
    joined_df = labels_df.join(ref_df, on='concept_id', how='left')

    # Select only the concept_name column and write to labels-name.tsv
    labels_name_df = joined_df.select('concept_name')
    labels_name_tsv_path = os.path.join(output_dir, 'labels-name.tsv')
    # We write a CSV and then replace commas with tabs
    labels_name_df.write_csv(labels_name_tsv_path)

    # As Polars writes with default comma separators, replace commas with tabs
    with open(labels_name_tsv_path, 'r') as f:
        content = f.read().replace(',', '\t')

    with open(labels_name_tsv_path, 'w') as f:
        f.write(content)

def convert_embedding_for_plp(input_filepath, output_filepath):
    old_state = torch.load(input_filepath, map_location='cpu')
    embeddings = old_state['state_dict']['embedding.weight'].to(dtype=torch.float32).cpu()
    concept_ids = torch.tensor(old_state['concept_ids'], dtype=torch.int64).cpu()
    new_state = {'concept_ids': concept_ids, 'embeddings': embeddings}
    torch.save(new_state, output_filepath)

