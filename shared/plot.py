#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared core for various hierarchy and embedding plots.
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
from pathlib import Path

# --------------------------------------------------------------------------- #
#                  OTHER IMPORTS                                              #
# --------------------------------------------------------------------------- #
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from IPython import display

# --------------------------------------------------------------------------- #
#                  OWN IMPORTS                                                #
# --------------------------------------------------------------------------- #
from shared.math import eucl_dist, poincare_dist
from shared.io import get_dict_data

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
def plot_hierarchy(data, objects, ref):
    """Plots a labeled hierarchy of nodes and edges.
    :param data: Pair list of edges
    :param objects: List of unique concept identifiers
    :param ref: List of unique concept labels
    :return:
    """
    # create a dictionary of objects and labels
    dict_data = get_dict_data(objects, ref)

    g = nx.Graph()
    g.add_edges_from(data)

    # set the hierarchy layout
    pos = nx.spring_layout(g)
    # pos = nx.kamada_kawai_layout(g)

    pos_attrs = {}
    for node, coords in pos.items():
        pos_attrs[node] = (coords[0], coords[1] + 0.08)

    # print(nx.info(g))
    plt.figure()
    nx.draw(g, pos=pos, node_size=10, node_color="lightgray", width=0.2)
    nx.draw_networkx_labels(g, labels=dict_data, pos=pos, font_size=4)
    plt.savefig(Path("output", "hierarchy.png"), dpi=300)
    plt.show()


def plot_geodesic():
    """Plots a grid of disks to visualize Poincaré and Euclidean distance.
    :return:
    """
    # plot unit circle in R^2
    fig = plt.figure(figsize=(20,12))
    ax = fig.add_subplot(2, 3, 1)

    # sample within unit circle in R^2
    n = 2000
    theta = np.random.uniform(0, 2*np.pi, n)
    u = np.random.uniform(0, 1, n)
    r = np.sqrt(u)
    X = np.array([r * np.cos(theta), r * np.sin(theta)]).T
    i = np.random.choice(n)

    xi = [0.0, 0.0]
    dist_xi = [poincare_dist(xi, x) for x in X]
    im = ax.scatter(X[:,0], X[:,1], s=20, c=dist_xi, cmap='inferno_r', edgecolors='white')
    ax.scatter(xi, xi, s =50, c='black')
    ax.set_title('Poincaré distance from ' + str(xi))
    plt.colorbar(im)

    ax = fig.add_subplot(2, 3, 2)
    xi = [0.5, 0.5]
    dist_xi = [poincare_dist(xi, x) for x in X]
    im = ax.scatter(X[:,0], X[:,1], s=20, c=dist_xi, cmap='inferno_r', edgecolors='white')
    ax.scatter(xi, xi, s =50, c='black')
    ax.set_title('Poincaré distance from ' + str(xi))
    plt.colorbar(im);

    ax = fig.add_subplot(2, 3, 3)
    xi = [0.7, 0.7]
    dist_xi = [poincare_dist(xi, x) for x in X]
    im = ax.scatter(X[:,0], X[:,1], s=20, c=dist_xi, cmap='inferno_r', edgecolors='white')
    ax.scatter(xi, xi, s =50, c='black')
    ax.set_title('Poincaré distance from ' + str(xi))
    plt.colorbar(im)

    ax = fig.add_subplot(2, 3, 4)
    xi = [0.0, 0.0]
    dist_xi = [eucl_dist(xi, x) for x in X]
    im = ax.scatter(X[:,0], X[:,1], s=20, c=dist_xi, cmap='inferno_r', edgecolors='white')
    ax.scatter(xi, xi, s =50, c='black')
    ax.set_title('Euclidean distance from ' + str(xi))
    plt.colorbar(im)

    ax = fig.add_subplot(2, 3, 5)
    xi = [0.5, 0.5]
    dist_xi = [eucl_dist(xi, x) for x in X]
    im = ax.scatter(X[:,0], X[:,1], s=20, c=dist_xi, cmap='inferno_r', edgecolors='white')
    ax.scatter(xi, xi, s =50, c='black')
    ax.set_title('Euclidean distance from ' + str(xi))
    plt.colorbar(im);

    ax = fig.add_subplot(2, 3, 6)
    xi = [0.7, 0.7]
    dist_xi = [eucl_dist(xi, x) for x in X]
    im = ax.scatter(X[:,0], X[:,1], s=20, c=dist_xi, cmap='inferno_r', edgecolors='white')
    ax.scatter(xi, xi, s =50, c='black')
    ax.set_title('Euclidean distance from ' + str(xi))
    plt.colorbar(im)

    plt.savefig(Path("output", "geodesic.png"), dpi=300)
    plt.show()


def plot_train_embed(model, mean_rank, epoch, epoch_loss):
    coordinates = model.embedding.weight.detach().numpy()
    # pl.annotate(dictData[x], (coordinates[x,0], coordinates[x,1]), fontsize=4)
    plt.figure()
    plt.gca().clear()
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title(f"Epoch: {epoch}, mean rank: {round(mean_rank, 3)}, last loss: {round(epoch_loss, 3)}")
    plt.scatter(coordinates[:,0], coordinates[:,1],s=1)
    display.clear_output(wait=True)
    display.display(plt.gcf())
# --------------------------------------------------------------------------- #
#                  LOCAL FUNCTIONS                                            #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#                  END OF FILE                                                #
# --------------------------------------------------------------------------- #
