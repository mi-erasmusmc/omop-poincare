#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Brief description to fit on single line.
Elaborate description spanning multiple lines, covering functionality in more
detail and explaining module usage."""
# --------------------------------------------------------------------------- #
#                  MODULE HISTORY                                             #
# --------------------------------------------------------------------------- #
# Version          1
# Date             2021-0728
# Author           LH John, E Fridgeirsson
# Note             Original version
#
# --------------------------------------------------------------------------- #
#                  SYSTEM IMPORTS                                             #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#                  OTHER IMPORTS                                              #
# --------------------------------------------------------------------------- #
import torch
from torch.nn import Embedding

# --------------------------------------------------------------------------- #
#                  OWN IMPORTS                                                #
# --------------------------------------------------------------------------- #
import riemannianSgd

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
class Model(torch.nn.Module):
    def __init__(self, dim, size, init_weights=1e-3, epsilon=1e-7):
        super().__init__()
        self.embedding = Embedding(size, dim, sparse=False)
        self.embedding.weight.data.uniform_(-init_weights, init_weights)
        self.epsilon = epsilon

    def dist(self, u, v):
        sqdist = torch.sum((u - v) ** 2, dim=-1)
        squnorm = torch.sum(u ** 2, dim=-1)
        sqvnorm = torch.sum(v ** 2, dim=-1)
        x = 1 + 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + self.epsilon
        z = torch.sqrt(x ** 2 - 1)
        return torch.log(x + z)

    def forward(self, inputs):
        e = self.embedding(inputs)
        o = e.narrow(dim=1, start=1, length=e.size(1) - 1)
        s = e.narrow(dim=1, start=0, length=1).expand_as(o)

        return self.dist(s, o)

# --------------------------------------------------------------------------- #
#                  LOCAL FUNCTIONS                                            #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
#                  END OF FILE                                                #
# --------------------------------------------------------------------------- #
