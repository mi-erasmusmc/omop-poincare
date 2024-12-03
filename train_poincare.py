import json
import pathlib
from argparse import Namespace

import networkx as nx
from numpy import savetxt
from tqdm import tqdm
from dataclasses import dataclass, fields

import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler, RandomSampler
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
import hypll.nn as hnn
from hypll.optim import RiemannianSGD, RiemannianAdam
import simple_parsing

torch.set_default_dtype(torch.float32)


@dataclass
class Args:
    # The path to the data file
    data_file: str = 'data/snomed_graph.json'
    # The path to a model file if we want to load a pre-trained model
    model_file: str = 'poincare_embeddings_snomed.pt'
    # output directory
    output_directory: str = './output/dim_50_negative_300_conditions_wd14'
    # The number of models to save
    save_top: int = 5
    # The number of epochs to train for
    epochs: int = 300
    # The learning rate
    learning_rate: float = 0.3
    # Whether to perform burn-in training.
    burn_in: bool = False
    # The learning rate divisor for burn-in [int]
    burn_in_lr_divisor: float = 10
    # The number of burn-in epochs [int]
    burn_in_epochs: int = 10
    # The batch size [int]
    batch_size: int = 128
    # The embedding dimension [int]
    embedding_dim: int = 50
    # The curvature of the Poincare ball [float]
    curvature: float = 0.542 # 0.542 to get curvature of 1
    # Whether to compile the model [bool]
    compile: bool = False
    # The device to use [str]
    device: str = 'cuda'
    # negative samples [int]
    negative_samples: int = 300
    # optimizer to use either 'adam' or 'sgd'
    optimizer: str = 'adam'


def save_model(model, args, output_directory, epoch, average_loss, loss_per_epoch):
    # only keep top args.save_top models
    saved_models = len(list(output_directory.glob(f"epoch:*-{args.model_file}")))
    while saved_models >= args.save_top:
        # find worst loss value from filename of saved models
        candidate_models = list(output_directory.glob(f"epoch:*-{args.model_file}"))
        losses = [float(model.name.split('-')[1].split('-')[0].split(':')[1]) for model in candidate_models]
        worst_loss = max(losses)
        worst_model = [model for model in candidate_models if float(model.name.split('-')[1].split('-')[0].split(':')[1]) == worst_loss][0]
        # remove the worst model
        if worst_model.exists():
            worst_model.unlink()
        saved_models -= 1
    torch.save({'state_dict': model.state_dict(),
            'args': args.__dict__, # convert dataclass to dict
            'losses': loss_per_epoch,
            'epoch': epoch,
            }, output_directory.joinpath(f"epoch:{epoch}-loss:{average_loss:3f}-{args.model_file}"))


def train_model(args: Namespace):
    dataset = GraphEmbeddingDataset(json_file=args.data_file,
                                    device= args.device,
                                    directed= False,
                                    negative_samples = args.negative_samples,
                                    domain="Condition")
    dataloader = DataLoader(dataset, sampler=BatchSampler(sampler=RandomSampler(dataset), batch_size=args.batch_size,
                                                          drop_last=False))
    output_directory = pathlib.Path(args.output_directory)
    if not output_directory.exists():
        output_directory.mkdir(parents=True)
    poincare_ball = PoincareBall(Curvature(args.curvature))

    model = PoincareEmbedding(
        num_embeddings=dataset.num_nodes,
        embedding_dim=args.embedding_dim,
        manifold=poincare_ball,
        root_concept_index=dataset.root_node
    )
    if args.compile:
        model = torch.compile(model, fullgraph=True)
    model = model.to(args.device)
    if args.optimizer == 'adam':
        optimizer_object = RiemannianAdam
    else:
        optimizer_object = RiemannianSGD

    if args.burn_in:
        optimizer = optimizer_object(
            params=model.parameters(),
            lr=args.learning_rate / args.burn_in_lr_divisor,
        )
        for epoch in range(args.burn_in_epochs):
            average_loss = torch.tensor(0.0, device=args.device)
            for idx, (edges, edge_label_targets) in tqdm(enumerate(dataloader)):
                optimizer.zero_grad()

                dists = model(edges)
                loss = poincare_embeddings_loss(distances=dists, targets=edge_label_targets)
                loss.backward()
                optimizer.step()

                average_loss += loss.detach()

            average_loss /= len(dataloader)
            print(f"Burn-in epoch {epoch} loss: {average_loss}")

    # Now we use the actual learning rate
    optimizer = optimizer_object(
        params=model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    early_stopper = EarlyStopper(patience=4)
    loss_per_epoch = torch.empty(args.epochs, device=args.device)
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(args.epochs):
        average_loss = torch.tensor(0.0, device=args.device)
        for idx, (edges, edge_label_targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
            optimizer.zero_grad()

            dists = model(edges)
            loss = poincare_original_loss(distances=dists, targets=edge_label_targets)
            loss.backward()
            optimizer.step()

            average_loss += loss.detach()
        average_loss /= len(dataloader)
        scheduler.step(average_loss)
        if average_loss < best_loss:
            best_loss = average_loss
            best_epoch = epoch
            save_model(model, args, output_directory, epoch, average_loss, loss_per_epoch)

        if early_stopper.update(average_loss):
            print(f"Early stopping at epoch {epoch}")
            break
        print(f"Epoch {epoch} loss: {average_loss} lr: {scheduler.get_last_lr()}")
        loss_per_epoch[epoch] = average_loss

    model.load_state_dict(torch.load(output_directory.joinpath(f"epoch:{best_epoch}-loss:{best_loss:3f}-{args.model_file}"))['state_dict'])

    mean_rank, map_score = evaluate_model(model, dataset)
    # save rank and map as txt file
    savetxt(output_directory.joinpath(f"mean-rank:{mean_rank}_map:{map_score}.txt"), [mean_rank, map_score])

def save_for_plp(embeddings, dataset, args):
    concept_ids = torch.as_tensor(list(nx.get_node_attributes(dataset.graph, 'concept_id').values()), dtype=torch.long)
    torch.save({'concept_ids': concept_ids, 'embeddings': embeddings}, 'poincare_embeddings_snomed_plp.pt')

def evaluate_model(model: torch.nn.Module, dataset, batch_size: int = 32):
    mean_rank, map_score = evaluate_mean_rank_and_map(dataset, model.weight.tensor.detach(), dataset.num_nodes,
                                                      batch_size=batch_size)
    print(f"Mean Rank: {mean_rank}, MAP: {map_score}")
    return mean_rank, map_score


class GraphEmbeddingDataset(Dataset):
    def __init__(
            self,
            json_file: str,
            device: torch.device = torch.device("cuda"),
            negative_samples: int = 10,
            directed: bool = False,
            domain: str = None):
        super().__init__()
        with open(json_file, "r") as json_file:
            graph_dict = json.load(json_file)

        graph = nx.node_link_graph(graph_dict, directed=directed)

        if domain is not None:
            self.domain = domain
            # filter graph
            graph = nx.subgraph(graph, [node for node in graph.nodes() if graph.nodes[node]['domain'] == domain
            and graph.nodes[node]['vocabulary'] == 'SNOMED'])
            # update node indexes
            mapping = {node: idx for idx, node in enumerate(graph.nodes())}
            graph = nx.relabel_nodes(graph, mapping)
            # root node has Clinical finding concept_name
            self.root_node = [node for node in graph.nodes() if graph.nodes[node]['concept_name'] == 'Clinical finding'][0]
        self.graph = graph
        self.edges_list = torch.as_tensor(list(graph.edges()), device=device)
        self.device = device
        self.num_nodes = graph.number_of_nodes()
        self.negative_samples = negative_samples
        self.directed = directed
        # self.adjacency_sets = self.generate_adjacency_sets()
        self.adjacency_matrix = self.create_sparse_adjacency_matrix()

    # This Dataset object has a sample for each edge in the graph.
    def __len__(self) -> int:
        return len(self.edges_list)

    def create_sparse_adjacency_matrix(self):
        values = torch.ones(size=(self.edges_list.shape[0],), dtype=torch.float32, device=self.device)
        adjacency_matrix = torch.sparse_coo_tensor(
            indices=self.edges_list.T,
            values=values,
            size=(self.num_nodes, self.num_nodes),
            device=self.device
        )
        if not self.directed:
            adjacency_matrix += adjacency_matrix.T
        adjacency_matrix = adjacency_matrix.coalesce()
        return adjacency_matrix

    def generate_adjacency_sets(self):
        adjacency_sets = [set() for _ in range(self.num_nodes)]
        directed = self.directed
        for edge in self.edges_list.cpu().numpy():
            u, v = edge
            adjacency_sets[u].add(v)
            if not directed:
                adjacency_sets[v].add(u)
        return adjacency_sets

    def sample_negative_uniform(self, batch_size: int):
        negative_nodes = torch.randint(low=0, high=self.num_nodes,
                                       size=(batch_size * self.negative_samples,),
                                       device=self.device)
        return negative_nodes


    def __getitem__(self, idx: int):
        # For each existing edge in the graph we choose 10 fake or negative edges, which we build
        # from the idx-th existing edge. So, first we grab this edge from the graph.
        batch_rel = self.edges_list[idx]
        source_nodes = batch_rel[:, 0]
        target_nodes = batch_rel[:, 1]
        batch_size = source_nodes.size(0)
        # Next, we take our source node rel[0] and see which nodes in the graph are not a child of
        # this node.
        negative_target_nodes =  self.sample_negative_uniform(
            batch_size=source_nodes.size(0))
        source_nodes_expanded = source_nodes.unsqueeze(1).expand(-1, self.negative_samples).reshape(-1)
        negative_edge_indices = torch.stack([source_nodes_expanded, negative_target_nodes], dim=0)

        negative_edge_values = self.adjacency_matrix._values()
        negative_edge_indices_in_adj = self.adjacency_matrix._indices()
        adj_edge_keys = negative_edge_indices_in_adj[0] * self.num_nodes + negative_edge_indices_in_adj[1]
        negative_edge_keys = negative_edge_indices[0] * self.num_nodes + negative_edge_indices[1]
        mask = ~torch.isin(negative_edge_keys, adj_edge_keys)

        valid_negative_source_nodes = source_nodes_expanded[mask]
        valid_negative_target_nodes = negative_target_nodes[mask]

        positive_edges = torch.stack([source_nodes, target_nodes], dim=1)
        negative_edges = torch.stack([valid_negative_source_nodes, valid_negative_target_nodes], dim=1)

        positive_labels = torch.ones(positive_edges.size(0), device=self.device)
        negative_labels = torch.zeros(negative_edges.size(0), device=self.device)

        edges = torch.cat([positive_edges, negative_edges], dim=0)
        labels = torch.cat([positive_labels, negative_labels], dim=0).bool()
        return edges, labels
        # # ensure negative samples are not the same as the source nodes
        # source_nodes_expanded = source_nodes.unsqueeze(1).expand(-1, self.negative_samples)
        # mask = negative_target_nodes != source_nodes_expanded
        # while not mask.all():
        #     resample_indices = torch.where(~mask)
        #     negative_target_nodes[resample_indices] = torch.randint(
        #         0, self.num_nodes, (resample_indices[0].size(0),), device=self.device
        #     )
        #     mask = negative_target_nodes != source_nodes_expanded

        # negative_source_nodes = source_nodes.unsqueeze(1).expand(-1, self.negative_samples)
        # negative_source_nodes = negative_source_nodes.contiguous().view(-1)
        # negative_target_nodes = negative_target_nodes.contiguous().view(-1)
        #
        # all_source_nodes = torch.cat([source_nodes, negative_source_nodes], dim=0)
        # all_target_nodes = torch.cat([target_nodes, negative_target_nodes], dim=0)
        #
        # edges = torch.stack([all_source_nodes, all_target_nodes], dim=1)
        # positive_labels = torch.ones(batch_size, device=self.device).bool()
        # negative_labels = torch.zeros(batch_size * self.negative_samples, device=self.device).bool()
        # labels = torch.cat([positive_labels, negative_labels], dim=0)
        # return edges, labels


class PoincareEmbedding(hnn.HEmbedding):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            manifold: PoincareBall,
            root_concept_index: int
    ):
        super().__init__(num_embeddings, embedding_dim, manifold)
        self.root_concept_index = root_concept_index

        if root_concept_index is not None:
            # Set the embedding of the root concept to zero to fix at origin
            with torch.no_grad():
                self.weight.data[self.root_concept_index].zero_()

            # register backward hook to zero out gradients of the root concept
            self.weight.tensor.register_hook(self._zero_root_grad)

    def _zero_root_grad(self, grad: torch.Tensor) -> torch.Tensor:
        grad[self.root_concept_index].zero_()
        return grad

    def forward(self, edges: torch.Tensor) -> torch.Tensor:
        embeddings = super().forward(edges)
        edge_distances = self.manifold.dist(x=embeddings[:, :, 0, :], y=embeddings[:, :, 1, :])
        return edge_distances


class EarlyStopper:
    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.tolerance = 1e-5

    def update(self, loss: float) -> bool:
        if loss < self.best_loss - self.tolerance:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def poincare_embeddings_loss(distances: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = distances.neg().exp()
    numerator = torch.where(condition=targets, input=logits, other=0).sum(dim=-1)
    denominator = logits.sum(dim=-1)
    loss = (numerator / denominator).log().mean().neg()
    return loss

def poincare_original_loss(distances: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    positive_loss = -torch.log(torch.sigmoid(-distances[targets]))
    negative_loss = -torch.log(torch.sigmoid(distances[~targets]))
    loss = (positive_loss.mean() + negative_loss.mean()) / 2
    return loss

def hyperbolic_distance(u, v, c=1.0, eps=1e-5):
    """
    Computes the Poincaré distance between two sets of points.

    Parameters:
    - u: Tensor of shape [batch_size, embedding_dim]
    - v: Tensor of shape [num_nodes, embedding_dim]

    Returns:
    - distances: Tensor of shape [batch_size, num_nodes], containing the distances between each u[i] and v[j].
    """
    # Calculate norms
    sqrt_c = c ** 0.5
    u_norm_sq = torch.sum(u ** 2, dim=1, keepdim=True)  # Shape: [batch_size, 1]
    v_norm_sq = torch.sum(v ** 2, dim=1, keepdim=True)  # Shape: [num_nodes, 1]

    # Ensure norms are within the ball
    u_norm_sq = torch.clamp(u_norm_sq, max=(1 - eps))
    v_norm_sq = torch.clamp(v_norm_sq, max=(1 - eps))

    # Compute inner product
    u = u.unsqueeze(1)  # Shape: [batch_size, 1, embedding_dim]
    v = v.unsqueeze(0)  # Shape: [1, num_nodes, embedding_dim]
    diff = u - v  # Shape: [batch_size, num_nodes, embedding_dim]
    diff_norm_sq = torch.sum(diff ** 2, dim=2)  # Shape: [batch_size, num_nodes]

    # Compute denominator
    denom = (1 - c * u_norm_sq) * (1 - c * v_norm_sq).transpose(0, 1)  # Shape: [batch_size, num_nodes]
    denom = denom.clamp(min=eps)

    # Compute the argument of arcosh
    argument = 1 + (2 * c * diff_norm_sq) / denom

    # Ensure argument is >= 1
    argument = torch.clamp(argument, min=1 + eps)

    # Compute the distance
    distances = torch.acosh(argument)

    return distances

def fast_average_precision(y_true, y_score):
    # Sort scores in descending order and get sorted indices
    desc_score_indices = torch.argsort(y_score, descending=True)
    y_true_sorted = y_true[desc_score_indices]

    num_positive = y_true.sum().item()
    if num_positive == 0:
        return 0.0
    # Calculate true positives and false positives
    tp_cumsum = torch.cumsum(y_true_sorted, dim=0)
    # Calculate precision at each position where y_true_sorted == 1
    positions = torch.arange(1, len(y_true_sorted) + 1, dtype=torch.float32, device=y_true.device)
    precision_at_k = tp_cumsum / positions

    # Only consider positions where y_true_sorted == 1
    precision_at_positives = precision_at_k[y_true_sorted == 1]
    ap = precision_at_positives.sum() / num_positive
    # Move the result back to CPU if needed and return a Python float
    return ap

def evaluate_mean_rank_and_map(dataset, embeddings, num_nodes, batch_size=128):
    edges_list = dataset.edges_list
    device = embeddings.device
    num_edges = edges_list.size(0)
    adjacency_sets = dataset.generate_adjacency_sets()
    mean_ranks = torch.empty(num_edges, dtype=torch.int64, device=device)
    average_precisions = torch.empty(num_edges, dtype=torch.float32, device=device)

    # Convert embeddings to double precision for numerical stability
    embeddings = embeddings.to(dtype=torch.double)

    for i in tqdm(range(0, num_edges, batch_size)):
        batch_edges = edges_list[i:i + batch_size]
        batch_size_actual = batch_edges.size(0)

        # Get source nodes and target nodes
        u_nodes = batch_edges[:, 0]  # Shape: [batch_size_actual]
        v_nodes = batch_edges[:, 1]  # Shape: [batch_size_actual]

        u_embeddings = embeddings[u_nodes]  # Shape: [batch_size_actual, embedding_dim]

        all_node_embeddings = embeddings

        distances =  hyperbolic_distance(u_embeddings, all_node_embeddings, c=1.0)

        mask = torch.zeros((batch_size_actual, num_nodes), dtype=torch.bool, device=device)

        for idx in range(batch_size_actual):
            u_int = u_nodes[idx].item()
            v_int = v_nodes[idx].item()
            connected_v = adjacency_sets[u_int]
            mask[idx, list(connected_v)] = True  # Mark connected nodes to exclude
            mask[idx, u_int] = True  # Exclude u_node itself
            mask[idx, v_int] = False  # Include the positive v_node

        valid_distances = distances.clone()
        valid_distances[mask] = float('-inf')  # Set masked positions to -inf for max computation
        max_distance_per_row, _ = valid_distances.max(dim=1, keepdim=True)  # Shape: [batch_size_actual, 1]
        max_distance_plus_one = max_distance_per_row + 1  # Shape: [batch_size_actual, 1]
        max_distance_expanded = max_distance_plus_one.expand(-1,
                                                             distances.size(1))  # Shape: [batch_size_actual, num_nodes]
        distances[mask] = max_distance_expanded[mask]

        # Now compute ranks
        # The lower the distance, the higher the rank (rank 1 is the smallest distance)
        # So we can sort distances and get the indices
        sorted_distances, sorted_indices = torch.sort(distances, dim=1)

        # For each example, find the rank of the positive v_node
        matches = (sorted_indices == v_nodes.unsqueeze(1))
        ranks = matches.nonzero()[:, 1] + 1  # rank positions start from 1
        mean_ranks[i:i + batch_size_actual] = ranks

        # For MAP, we need to compute average precision for each example
        # Create labels: 1 for positive v_node, 0 for others
        labels = torch.zeros_like(distances, dtype=torch.int64, device=device)
        labels[torch.arange(batch_size_actual), v_nodes] = 1
        # After sorting, obtain sorted labels
        sorted_labels = torch.gather(labels, 1, sorted_indices)
        # Compute average precision for each u_node
        aps = torch.empty(batch_size_actual, dtype=torch.float32, device=device)
        for idx in range(batch_size_actual):
            ap = fast_average_precision(sorted_labels[idx], -sorted_distances[idx])
            aps[idx] = ap
        average_precisions[i:i + batch_size_actual] = aps

    # Calculate Mean Rank and MAP
    mean_rank = mean_ranks.float().mean().item()
    map_score = average_precisions.mean().item()
    return mean_rank, map_score


def save_plp_format(model_file: str):
    model = torch.load(model_file, map_location='cpu')
    embeddings = model['state_dict']['weight'].tensor

    torch.save()


if __name__ == "__main__":
    args = simple_parsing.parse(Args)

    train_model(args=args)
