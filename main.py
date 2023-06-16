
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
import csv
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --------------------------------------------------------------------------- #
#                  OWN IMPORTS                                                #
# --------------------------------------------------------------------------- #
from shared.plot import plot_geodesic, plot_hierarchy, plot_train_embed,\
    get_dict_data
from shared.io import read_data, read_ref
from shared.math import eval_reconstruction
from train import init_torch_objects, train

# --------------------------------------------------------------------------- #
#                  META DATA                                                  #
# --------------------------------------------------------------------------- #
__status__ = 'Development'

# --------------------------------------------------------------------------- #
#                  CONSTANTS                                                  #
# --------------------------------------------------------------------------- #
# OUT_DIMENSIONS = 2
# NEG_SAMPLES = 5
# EPOCH = 25

# --------------------------------------------------------------------------- #
#                  GLOBAL VARIABLES                                           #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#                  EXPORTED FUNCTIONS                                         #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#                  LOCAL FUNCTIONS                                            #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#                  END OF FILE                                                #
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    OUT_DIMENSIONS = 20
    NEG_SAMPLES = 50
    EPOCH = 300
    torch.set_default_dtype(torch.float64)

    # %%

    # Plot geodesic comparison between Poincaré and Euclidean
    # plot_geodesic()

    # %%

    # Load edge data
    data_path = Path("data", "dist1.csv")
    data, weights, objects, neighbors, diff_summed, num_relations = read_data(data_path)

    # load concept reference
    ref_path = Path('data', 'ref.csv')
    ref = read_ref(ref_path)

    # initialize torch objects for the training loop
    model, optimizer, loss_func = init_torch_objects(objects, OUT_DIMENSIONS)
    model = model.to("cuda:0")
    # ToDo: implement function to load embedding and continue training

    train(data=data, weights=weights, objects=objects, neighbors=neighbors,
          diff_summed=diff_summed, num_relations=num_relations,
          model=model, optimizer=optimizer, loss_func=loss_func,
          out_dimensions=OUT_DIMENSIONS, n_neg_samples=NEG_SAMPLES, n_epochs=EPOCH,
          n_burn_in=40)