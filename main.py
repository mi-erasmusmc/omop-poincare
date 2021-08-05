#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Brief description to fit on single line.
Elaborate description spanning multiple lines, covering functionality in more
detail and explaining module usage."""
# --------------------------------------------------------------------------- #
#                  MODULE HISTORY                                             #
# --------------------------------------------------------------------------- #
# Version          1
# Date             2021-07-28
# Author           LH John, E Fridgeirsson
# Note             Original version
#
# --------------------------------------------------------------------------- #
#                  SYSTEM IMPORTS                                             #
# --------------------------------------------------------------------------- #
from pathlib import Path

# --------------------------------------------------------------------------- #
#                  OTHER IMPORTS                                              #
# --------------------------------------------------------------------------- #
import torch
from torch.distributions import Categorical
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# --------------------------------------------------------------------------- #
#                  OWN IMPORTS                                                #
# --------------------------------------------------------------------------- #
from riemannian_sgd import RiemannianSGD
from model import Model
import shared

# --------------------------------------------------------------------------- #
#                  META DATA                                                  #
# --------------------------------------------------------------------------- #
__status__ = 'Development'

# --------------------------------------------------------------------------- #
#                  CONSTANTS                                                  #
# --------------------------------------------------------------------------- #
OUT_DIMENSIONS = 2
NEG_SAMPLES = 5
EPOCH = 25

# --------------------------------------------------------------------------- #
#                  GLOBAL VARIABLES                                           #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#                  EXPORTED FUNCTIONS                                         #
# --------------------------------------------------------------------------- #
data, objects, weights = shared.load_edge_list(Path("data", "dist1Sample.csv"))
print(data.shape)
print(objects)
# data = BatchedDataset(idx, objects, weights, opt.negs, opt.batchsize,
#                       opt.ndproc, opt.burnin > 0, opt.dampening)

# --------------------------------------------------------------------------- #
#                  LOCAL FUNCTIONS                                            #
# --------------------------------------------------------------------------- #
# data = helpers.read_data(Path("data", "dist1Sample.csv"))
#print(data)

cat_dist = Categorical(probs=torch.from_numpy(weights)) # was weights
#print(cat_dist)
unif_dist = Categorical(probs=torch.ones(data.shape[0],) / data.shape[0])

model = Model(dim=OUT_DIMENSIONS, size=data.shape[0])
optimizer = RiemannianSGD(model.parameters())

loss_func = CrossEntropyLoss()
batch_X = torch.zeros(10, NEG_SAMPLES + 2, dtype=torch.long)
batch_y = torch.zeros(10, dtype=torch.long)

while True:
    if EPOCH < 20:
        lr = 0.003
        sampler = cat_dist
    else:
        lr = 0.3
        sampler = unif_dist

    perm = torch.randperm(data.shape[0])
    dataset_rnd = data.loc[perm, ]

    for i in tqdm(range(0, data.shape[0] - data.shape[0] % 10, 10)):
        batch_X[:, :2] = dataset_rnd[i: i + 10]

        for j in range(10):
            a = set(sampler.sample([2 * NEG_SAMPLES]).numpy())
            negatives = list(a - (set(neighbors[batch_X[j, 0]]) | set(neighbors[batch_X[j, 1]])))
            batch_X[j, 2 : len(negatives)+2] = torch.LongTensor(negatives[:NEG_SAMPLES])

        optimizer.zero_grad()
        preds = model(batch_X)

        loss = loss_func(preds.neg(), batch_y)
        loss.backward()
        optimizer.step(lr=lr)

# --------------------------------------------------------------------------- #
#                  END OF FILE                                                #
# --------------------------------------------------------------------------- #