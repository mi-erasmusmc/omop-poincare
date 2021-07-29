import pandas as pd

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