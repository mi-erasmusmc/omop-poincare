#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training loop for Poincaré embedding.
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

# --------------------------------------------------------------------------- #
#                  OTHER IMPORTS                                              #
# --------------------------------------------------------------------------- #
import torch
from tqdm import tqdm

# --------------------------------------------------------------------------- #
#                  OWN IMPORTS                                                #
# --------------------------------------------------------------------------- #
from model import Model
from riemannian_sgd import RiemannianSGD
from shared.math import eval_reconstruction

# --------------------------------------------------------------------------- #
#                  META DATA                                                  #
# --------------------------------------------------------------------------- #
__status__ = 'Development'

# --------------------------------------------------------------------------- #
#                  CONSTANTS                                                  #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#                  GLOBAL VARIABLES                                           #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#                  EXPORTED FUNCTIONS                                         #
# --------------------------------------------------------------------------- #
def train(data, weights, objects, neighbors, diff_summed, num_relations,
          model, optimizer, loss_func,
          out_dimensions, n_neg_samples, n_epochs, n_burn_in=40):

    # initialize some additional (temporary) objects for the training loop
    batch_X, batch_y, cat_dist, unif_dist = __init_data_objects(objects,
                                                                weights,
                                                                n_neg_samples)

    epoch_loss = 0.0
    mean_rank = 0.0
    last_loss = 0.0
    n = 0
    while n < n_epochs:
        print(f"Epoch: {n}")
        if n < n_burn_in:
            lr = 0.003
            sampler = cat_dist
        else:
            lr = 0.3
            sampler = unif_dist

        perm = torch.randperm(data.shape[0])
        # dataset_rnd = data.loc[perm, ]
        dataset_rnd = torch.as_tensor(data[perm, ])

        for i in tqdm(range(0, data.shape[0] - data.shape[0] % 10, 10)):
            batch_X[:, :2] = dataset_rnd[i: i + 10]

            for j in range(10):
                negatives = sampler.sample([2 * n_neg_samples]).unique(sorted=False)
                negatives = negatives[(negatives != batch_X[j, 0]) & (negatives != batch_X[j, 1])]
                batch_X[j, 2:negatives.size(0)+2] = negatives[:n_neg_samples]

                # Alternative implementation of constructing batch
                # a = set(sampler.sample([2 * NEG_SAMPLES]).numpy())
                # negatives = list(
                #     a - (set(data[batch_X[j, 0]]) | set(data[batch_X[j, 1]])))
                # batch_X[j, 2:len(negatives) + 2] = torch.LongTensor(
                #     negatives[:NEG_SAMPLES])

            optimizer.zero_grad()
            preds = model(batch_X)

            loss = loss_func(preds.neg(), batch_y)
            loss.backward()
            optimizer.step(lr=lr)

            # rank and loss output
            epoch_loss += loss.item()
        if n % 1 == 0:
            with torch.no_grad():
                mean_rank = eval_reconstruction(model, num_relations, neighbors, diff_summed)
            epoch_loss /= data.shape[0] // 10
            print(f"Mean rank: {mean_rank}, loss: {epoch_loss}")

            if mean_rank < last_loss or last_loss == 0:
                last_loss = mean_rank
                stop = 0
                state = {
                    "epoch": n,
                    "state_dict": model.state_dict(),
                    "mean_rank": mean_rank,
                    "loss": epoch_loss,
                    "names": objects
                }
                torch.save(state, f"output/poincare_model_dim_{out_dimensions}.pt")
            else:
                stop += 1

        n = n+1
    # reset counter to allow continuation of training
    return model


def init_torch_objects(objects, out_dimensions):
    model = Model(dim=out_dimensions, size=len(objects))
    optimizer = RiemannianSGD(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()
    return model, optimizer, loss_func

# --------------------------------------------------------------------------- #
#                  LOCAL FUNCTIONS                                            #
# --------------------------------------------------------------------------- #

def __init_data_objects(objects, weights, neg_samples):
    cat_dist = torch.distributions.Categorical(probs=torch.from_numpy(weights))
    # particularly important for  hyperboloid embedding stability
    unif_dist = torch.distributions.Categorical(
        probs=torch.ones(len(objects), ) / len(objects))
    batch_X = torch.zeros(10, neg_samples + 2, dtype=torch.long)
    batch_y = torch.zeros(10, dtype=torch.long)
    return batch_X, batch_y, cat_dist, unif_dist

# --------------------------------------------------------------------------- #
#                  END OF FILE                                                #
# --------------------------------------------------------------------------- #
