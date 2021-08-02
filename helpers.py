import pandas as pd
from tqdm import tqdm
from scipy.sparse import lil_matrix
import numpy as np

def read_data(data):
    return pd.read_csv(data, sep=',')

def load_edge_list(path, symmetrize=False):
    df = pd.read_csv(path, usecols=['id1', 'id2', 'weight'], engine='c')
    df.dropna(inplace=True)
    if symmetrize:
        rev = df.copy().rename(columns={'id1' : 'id2', 'id2' : 'id1'})
        df = pd.concat([df, rev])
    idx, objects = pd.factorize(df[['id1', 'id2']].values.reshape(-1))
    idx = idx.reshape(-1, 2).astype('int')
    weights = df.weight.values.astype('float')
    return idx, objects.tolist(), weights

def load_dataset(path, skip_stats=True, remove_duplicates=False):
    print("Processing dataset...")

    if "hyperlex" in path:
        df = pd.read_csv(path, usecols=[0, 1, 3], engine='c', delim_whitespace=True)
    else:
        df = pd.read_csv(path, usecols=[0, 1], engine='c', escapechar='\\', na_filter=False)

    ids, names = pd.factorize(df.iloc[:,:2].values.reshape(-1))
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

    print(f"Dataset size: {df.shape[0]}")
    print(f"Relations: {relations}")
    print(f"Vocabulary size: {voc_size}")
    #if relations < 10000:
    #    compute_hyperbolicity_and_other_stats(adjacency_matrix)

    if remove_duplicates:
        ids = np.array(list(set([tuple(x) for x in ids.tolist()])))

    #print(f"Hyperbolicity sample: {compute_avg_hyperbolicity_sample(adjacency_matrix)}")

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