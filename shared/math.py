#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared core for mathematical operations.
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
import numpy as np
import torch
from numba import jit
from tqdm import tqdm

# --------------------------------------------------------------------------- #
#                  OWN IMPORTS                                                #
# --------------------------------------------------------------------------- #

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
def lorentzian_inner_product(u, v, keepdim=False):
    uv = u * v
    uv.narrow(-1, 0, 1).mul_(-1)
    return torch.sum(uv, dim=-1, keepdim=keepdim)

def norm(x, axis=None):
    return np.linalg.norm(x, axis=axis)

def poincare_dist(u, v, eps=1e-5):
    d = 1 + 2 * norm(u-v)**2 / ((1 - norm(u)**2) * (1 - norm(v)**2) + eps)
    return np.arccosh(d)

def eucl_dist(u, v):
    return np.linalg.norm(u-v)

@jit(nopython=True)
def in_array(neighbors, sorted_idx):
    rank_sum = 0
    for k in neighbors:
        for q, k2 in enumerate(sorted_idx):
            if k == k2:
                rank_sum += q + 1
                break
    return rank_sum


def eval_reconstruction(model, num_relations, neighbors, diff_summed):
    """Calculate the mean rank of an embedding.
    :param model:
    :param num_relations:
    :param neighbors:
    :param diff_summed:
    :return:
    """
    voc_size = model.embedding.weight.size(0)
    rank_sum = 0.0
    print("Evaluating mean rank:")
    for i in tqdm(range(voc_size)):
        dists = model.dist(model.embedding.weight[None, i], model.embedding.weight)
        dists[i] = 1e12
        sorted_idx = dists.argsort()
        rank_sum += in_array(neighbors[i], sorted_idx.numpy())
    return (rank_sum - diff_summed) / num_relations


@torch.jit.script
def lambda_x(x: torch.Tensor):
    return 2 / (1 - torch.sum(x ** 2, dim=-1, keepdim=True))


@torch.jit.script
def mobius_add(x: torch.Tensor, y: torch.Tensor):
    x2 = torch.sum(x ** 2, dim=-1, keepdim=True)
    y2 = torch.sum(y ** 2, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    denom = 1 + 2 * xy + x2 * y2
    return num / denom.clamp_min(1e-15)


@torch.jit.script
def expm(p: torch.Tensor, u: torch.Tensor):
    return p + u
    # for exact exponential mapping
    #norm = torch.sqrt(torch.sum(u ** 2, dim=-1, keepdim=True))
    #return mobius_add(p, torch.tanh(0.5 * lambda_x(p) * norm) * u / norm.clamp_min(1e-15))


@torch.jit.script
def grad(p: torch.Tensor):
    p_sqnorm = torch.sum(p.data ** 2, dim=-1, keepdim=True)
    return p.grad.data * ((1 - p_sqnorm) ** 2 / 4).expand_as(p.grad.data)
