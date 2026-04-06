"""
build_signature_graph.py — Convert brainmaps.h5 into a PyG signature graph

Loads all brain maps from brainmaps.h5, computes the mean brain map signature
for each token, builds edges based on Jaccard overlap of active vertices,
and saves the resulting graph as signature_graph.pt.

Graph structure:
  - Nodes: one per token in vocabulary
  - Node features: mean brain map activation [N_VERTICES]
  - Edges: Jaccard overlap of active vertex sets > threshold
  - Labels: Jaccard overlap matrix (multi-label soft targets for training)

Usage:
    python build_signature_graph.py                                    # defaults
    python build_signature_graph.py --activation-thresh 0.1            # tune
    python build_signature_graph.py --jaccard-thresh 0.15 --input brainmaps.h5
"""

import argparse
import logging

import h5py
import numpy as np
import torch
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_brainmaps(input_path: str) -> tuple[list[str], np.ndarray]:
    """Load brain maps from HDF5 and compute mean signatures.

    Returns:
        token_list: list of token strings
        signatures: np.ndarray of shape [N_tokens, N_vertices]
    """
    token_list = []
    signatures = []

    with h5py.File(input_path, "r") as f:
        words_grp = f["words"]
        for grp_name in sorted(words_grp.keys()):
            grp = words_grp[grp_name]
            token = grp.attrs.get("word", grp_name)
            brain_map = grp["brain_map"][:]  # [T, N_VERTICES]

            # Mean across time steps → single signature vector
            signature = brain_map.mean(axis=0)  # [N_VERTICES]

            token_list.append(token)
            signatures.append(signature)

    signatures = np.stack(signatures, axis=0)  # [N_tokens, N_vertices]
    logger.info(f"Loaded {len(token_list)} tokens, signature shape: {signatures.shape}")
    return token_list, signatures


def compute_active_sets(signatures: np.ndarray, threshold: float) -> list[set]:
    """Compute the set of 'active' vertices for each token signature.

    A vertex is active if its activation exceeds the threshold.
    """
    active_sets = []
    for i in range(signatures.shape[0]):
        active = set(np.where(signatures[i] > threshold)[0])
        active_sets.append(active)
    return active_sets


def compute_jaccard_matrix(active_sets: list[set]) -> np.ndarray:
    """Compute pairwise Jaccard similarity between all token signatures.

    jaccard(A, B) = |A ∩ B| / |A ∪ B|
    """
    n = len(active_sets)
    jaccard = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i, n):
            if len(active_sets[i]) == 0 and len(active_sets[j]) == 0:
                jaccard[i, j] = 0.0
            else:
                intersection = len(active_sets[i] & active_sets[j])
                union = len(active_sets[i] | active_sets[j])
                if union > 0:
                    jaccard[i, j] = intersection / union
                else:
                    jaccard[i, j] = 0.0
            jaccard[j, i] = jaccard[i, j]

    return jaccard


def build_edges(jaccard_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """Build edge list from Jaccard matrix.

    An edge (i, j) exists if jaccard(i, j) > threshold and i != j.
    Returns edge_index of shape [2, E].
    """
    n = jaccard_matrix.shape[0]
    sources, targets = [], []

    for i in range(n):
        for j in range(i + 1, n):
            if jaccard_matrix[i, j] > threshold:
                # Add both directions (undirected graph)
                sources.extend([i, j])
                targets.extend([j, i])

    edge_index = np.array([sources, targets], dtype=np.int64)  # [2, E]
    return edge_index


def build_graph(token_list: list[str], signatures: np.ndarray,
                jaccard_matrix: np.ndarray, edge_index: np.ndarray,
                train_ratio: float = 0.6, val_ratio: float = 0.2) -> Data:
    """Build the PyG Data object for the signature graph.

    Graph:
      data.x          [N, N_VERTICES]   brain map feature vectors
      data.edge_index  [2, E]           Jaccard-overlap edges
      data.y           [N]              token index (self-supervised)
      data.jaccard     [N, N]           full Jaccard matrix (training target)
      data.train_mask  [N]              60% of vocabulary
      data.val_mask    [N]              20%
      data.test_mask   [N]              20%
      data.token_list  list[str]        token strings
    """
    n = len(token_list)

    # Create train/val/test masks
    indices = np.random.permutation(n)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[indices[:train_end]] = True
    val_mask[indices[train_end:val_end]] = True
    test_mask[indices[val_end:]] = True

    data = Data(
        x=torch.tensor(signatures, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.arange(n, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    # Store additional attributes
    data.jaccard = torch.tensor(jaccard_matrix, dtype=torch.float32)
    data.token_list = token_list
    data.num_tokens = n

    return data


def verify_graph(data: Data, token_list: list[str], jaccard_matrix: np.ndarray):
    """Print verification info about the graph."""
    print(f"\n{'='*60}")
    print(f"Signature Graph Summary")
    print(f"{'='*60}")
    print(f"  Nodes (tokens):     {data.num_nodes}")
    print(f"  Edges:              {data.num_edges}")
    print(f"  Features/node:      {data.x.shape[1]}")
    print(f"  Train nodes:        {int(data.train_mask.sum())}")
    print(f"  Val nodes:          {int(data.val_mask.sum())}")
    print(f"  Test nodes:         {int(data.test_mask.sum())}")
    print(f"  Avg degree:         {data.num_edges / data.num_nodes:.1f}")

    # Show cosine similarities for key pairs
    def cosine(a, b):
        a_np = a.numpy() if isinstance(a, torch.Tensor) else a
        b_np = b.numpy() if isinstance(b, torch.Tensor) else b
        dot = np.dot(a_np, b_np)
        norm = np.linalg.norm(a_np) * np.linalg.norm(b_np)
        return dot / norm if norm > 0 else 0.0

    print(f"\n  Key Similarity Checks:")
    test_pairs = [
        ("1+1", "2"),
        ("two", "2"),
        ("1+1", "3"),
        ("5+5", "10"),
    ]
    token_to_idx = {t: i for i, t in enumerate(token_list)}
    for t1, t2 in test_pairs:
        if t1 in token_to_idx and t2 in token_to_idx:
            i, j = token_to_idx[t1], token_to_idx[t2]
            cos = cosine(data.x[i], data.x[j])
            jac = jaccard_matrix[i, j]
            print(f"    {t1:10s} ↔ {t2:10s}  cosine={cos:.4f}  jaccard={jac:.4f}")

    # Show highest-connected nodes
    degrees = torch.zeros(data.num_nodes, dtype=torch.long)
    for i in range(data.edge_index.shape[1]):
        degrees[data.edge_index[0, i]] += 1
    top_k = min(10, data.num_nodes)
    top_indices = degrees.argsort(descending=True)[:top_k]
    print(f"\n  Most connected tokens:")
    for idx in top_indices:
        idx = idx.item()
        print(f"    {token_list[idx]:20s}  degree={degrees[idx].item()}")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Build signature graph from brain maps")
    parser.add_argument("--input", type=str, default="brainmaps.h5",
                        help="Input brain maps HDF5 file")
    parser.add_argument("--output", type=str, default="signature_graph.pt",
                        help="Output PyG graph file")
    parser.add_argument("--activation-thresh", type=float, default=0.1,
                        help="Vertex activation threshold (default: 0.1)")
    parser.add_argument("--jaccard-thresh", type=float, default=0.15,
                        help="Minimum Jaccard overlap for edges (default: 0.15)")
    parser.add_argument("--train-ratio", type=float, default=0.6,
                        help="Training set ratio (default: 0.6)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation set ratio (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val/test split")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Step 1: Load brain maps
    logger.info(f"Loading brain maps from {args.input}...")
    token_list, signatures = load_brainmaps(args.input)

    # Step 2: Compute active vertex sets
    logger.info(f"Computing active vertex sets (threshold={args.activation_thresh})...")
    active_sets = compute_active_sets(signatures, args.activation_thresh)

    active_counts = [len(s) for s in active_sets]
    logger.info(f"  Active vertices per token: "
                f"min={min(active_counts)}, max={max(active_counts)}, "
                f"mean={np.mean(active_counts):.0f}")

    # Step 3: Compute Jaccard similarity matrix
    logger.info("Computing pairwise Jaccard similarities...")
    jaccard_matrix = compute_jaccard_matrix(active_sets)

    # Step 4: Build edges
    logger.info(f"Building edges (Jaccard threshold={args.jaccard_thresh})...")
    edge_index = build_edges(jaccard_matrix, args.jaccard_thresh)
    logger.info(f"  Edges: {edge_index.shape[1]}")

    # Step 5: Build PyG graph
    logger.info("Building PyG Data object...")
    data = build_graph(
        token_list, signatures, jaccard_matrix, edge_index,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio,
    )

    # Step 6: Save
    torch.save(data, args.output)
    logger.info(f"Saved graph to {args.output}")

    # Step 7: Verify
    verify_graph(data, token_list, jaccard_matrix)


if __name__ == "__main__":
    main()
