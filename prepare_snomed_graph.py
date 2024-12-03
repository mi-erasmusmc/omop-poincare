import json

import networkx as nx
import polars as pl

# load data
nodes = pl.scan_ipc("/home/egill/github/omop-poincare/data/snomed_nodes.arrow").collect()
edges = pl.scan_ipc("/home/egill/github/omop-poincare/data/snomed_edges.arrow").collect()

all_concept_ids = nodes.select('concept_id').unique().with_row_index(name='id')

nodes = nodes.join(all_concept_ids, on='concept_id')

edges = (edges.with_columns(pl.col('ancestor_concept_id').alias('concept_id')).join(all_concept_ids, on='concept_id')
         .rename({"id": "ancestor_id"}).drop('concept_id').with_columns(pl.col('descendant_concept_id').alias('concept_id'))
         .join(all_concept_ids, on='concept_id').drop('concept_id')
         .rename({"id": "descendant_id"})
         .select(['ancestor_id', 'descendant_id']))

edge_list = list(zip(edges['ancestor_id'], edges['descendant_id']))

G = nx.DiGraph()
G.add_edges_from(edge_list)
G.add_nodes_from(nodes['id'])
node_attributes = nodes.select(['id',
                                'concept_id',
                                'concept_name',
                                'domain_id',
                                'vocabulary_id',
                                'concept_code',
                                'concept_class_id']).to_dicts()
node_attributes = {item['id']: {'concept_id': item['concept_id'],
                                'concept_name': item['concept_name'],
                                'domain': item['domain_id'],
                                'vocabulary': item['vocabulary_id'],
                                'concept_code': item['concept_code'],
                                'concept_class': item['concept_class_id']}
                   for item in node_attributes}
nx.set_node_attributes(G, node_attributes)

data = nx.readwrite.node_link_data(G)

output_path = "/home/egill/github/omop-poincare/data/snomed_graph.json"
with open(output_path, 'w') as f:
    json.dump(data, f, indent=4)

