import json
import pathlib

import networkx as nx
import numpy as np
import plotly.graph_objs as go
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Turbo256
from bokeh.plotting import figure
from plotly.offline import plot
from train_poincare import Args, GraphEmbeddingDataset
import torch
import umap
from umap.distances import hyperboloid_grad

# checkpoint_file = "/home/egill/github/omop-poincare/output/dim_50_negative_300_conditions_wd14/epoch:60-loss:0.665115-poincare_embeddings_snomed.pt"
checkpoint_file = "/home/egill/github/omop-poincare/output/dim_2_negative_10_conditions_curvature_wd14/epoch:24-loss:0.672215-poincare_embeddings_snomed.pt"
model = torch.load(checkpoint_file,
                   map_location=torch.device('cpu'))
embeddings =  model['state_dict']['weight'].tensor.numpy()

# load graph
dataset = GraphEmbeddingDataset(json_file="/home/egill/github/omop-poincare/data/snomed_graph_groups.json",
                                directed= True,
                                device= torch.device('cpu'),
                                negative_samples= 10,
                                domain='Condition')


def clean_embeddings(embeddings):
    json_file = "/home/egill/github/omop-poincare/data/snomed_graph.json"
    with open(json_file, 'r') as f:
        data = json.load(f)
    graph = nx.node_link_graph(data)
    domain = 'Condition'
    graph = nx.subgraph(graph, [node for node in graph.nodes() if graph.nodes[node]['domain'] == domain])
    # remap
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    graph = nx.relabel_nodes(graph, mapping)
    wrong_nodes = [node for node in graph.nodes() if graph.nodes[node]['vocabulary'] != 'SNOMED']
    # remove rows from embeddings
    embeddings = np.delete(embeddings, wrong_nodes, axis=0)
    return embeddings

def bokeh_scatter(embeddings, dataset):
    # Ensure the number of embeddings matches the number of nodes in the graph
    node_ids = list(dataset.graph.nodes)
    if embeddings.shape[0] != len(node_ids):
        embeddings = clean_embeddings(embeddings)

    # Extract node attributes in order matching embeddings
    concept_names = []
    group_numbers = []
    group_names = []

    for node_id in node_ids:
        node_data = dataset.graph.nodes[node_id]
        concept_names.append(node_data.get('concept_name', 'Unknown'))
        group_info = node_data.get('group', {'number': -1, 'name': 'Unknown'})
        group_numbers.append(group_info['number'])
        group_names.append(group_info['name'])

    # Map group numbers to colors
    unique_group_numbers = sorted(set(group_numbers))
    n_groups = len(unique_group_numbers)
    # Generate a palette with as many colors as there are groups
    indices = np.linspace(0, 255, n_groups).astype(int)
    palette = [Turbo256[i] for i in indices]
    group_number_to_color = dict(zip(unique_group_numbers, palette))
    colors = [group_number_to_color[gn] for gn in group_numbers]

    # Create the ColumnDataSource
    source = ColumnDataSource(data=dict(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        concept_name=concept_names,
        group_number=group_numbers,
        group_name=group_names,
        index=np.arange(embeddings.shape[0]),
        color=colors,
    ))

    p = figure(
        title='2D Poincaré Embeddings with Hover Interactivity',
        match_aspect=True,
        tools='pan,wheel_zoom,reset,save',
        output_backend='webgl'  # Enable WebGL for better performance
    )# Add circle (Poincaré disk boundary)
    p.circle(x=0, y=0, radius=1, fill_color=None, line_color='black')
    # Plot the embeddings
    # glyph = p.circle(
    #     'x', 'y', source=source,
    #     color='blue', size=2, alpha=0.6, line_color=None
    # )
    glyph = p.scatter(
        'x', 'y', source=source,
        color='color', size=2, alpha=0.6
    )

    # Add HoverTool
    hover = HoverTool(
        tooltips=[
            ('Index', '@index'),
            ('(x, y)', '(@x{0.000}, @y{0.000})'),
            ('Concept Name', '@concept_name'),
            ('Group Number', '@group_number'),
            ('Group Name', '@group_name'),
        ],
        renderers=[glyph]
    )
    p.add_tools(hover)
    output_file('poincare_embeddings_bokeh.html')
    show(p)

def plotly_scatter(embeddings, data):
    graph = nx.node_link_graph(data)
    hover_text = [graph.nodes[i]['concept_name'] for i in range(embeddings.shape[0])]

    # Create a Scattergl plot
    scatter = go.Scattergl(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        mode='markers',
        marker=dict(
            size=3,
            # color=radii,  # Optional: color points based on radii or another attribute
            colorscale='Viridis',
            showscale=True,
            opacity=0.7
        ),
        text=hover_text,
        hoverinfo='text'
    )

    # Unit circle (Poincaré disk boundary)
    theta = np.linspace(0, 2 * np.pi, 500)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    # Create circle trace
    circle = go.Scattergl(
        x=circle_x,
        y=circle_y,
        mode='lines',
        line=dict(color='black'),
        hoverinfo='skip'
    )

    # Create the figure
    fig = go.Figure(data=[scatter, circle])

    # Update layout
    fig.update_layout(
        title='2D Poincaré Embeddings with Hover Interactivity',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1.1, 1.1],
            scaleanchor='y',
            scaleratio=1
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1.1, 1.1]
        ),
        hovermode='closest',
        showlegend=False
    )

    # Save and open the plot in a browser
    plot(fig, filename='poincare_embeddings_plotly.html', auto_open=True)

def fit_umap(embeddings):
    embeddings = umap.UMAP(metric=hyperboloid_grad, verbose=True, init='pca').fit_transform(embeddings)
    # convert to 3d
    x = embeddings[:, 0]  # Corresponds to x_1
    y = embeddings[:, 1]  # Corresponds to x_2
    z = np.sqrt(1 + x ** 2 + y ** 2)  # Computes x_0

    # convert to poincare
    disk_x = x / (z + 1)  # Corresponds to u_1
    disk_y = y / (z + 1)  # Corresponds to u_2
    embeddings = np.vstack((disk_x, disk_y)).T

    return embeddings



# plotly_scatter(embeddings, data)
if embeddings.shape[1] > 2:
    embeddings = fit_umap(embeddings)
bokeh_scatter(embeddings, dataset)
