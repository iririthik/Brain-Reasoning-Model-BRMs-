"""
predict.py — Brain Signature Decoder: interactive prediction

Decodes ANY input by generating its brain map on-the-fly, injecting it
into the trained signature graph, running the GNN, and reading out which
token signatures are active.

Example:
    > 1+1
    Active signatures:
      "1"    0.94  ← surface token
      "+"    0.91  ← surface token
      "2"    0.87  ← resolved answer
      "two"  0.68  ← synonym via graph
      "="    0.71  ← equality concept

Usage:
    python predict.py                        # interactive mode
    python predict.py --word "1+1"           # single query
    python predict.py --word "5+3"           # decode any expression
    python predict.py --evaluate             # batch test on vocab
    python predict.py --mode cora            # standard CORA mode
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Import the brain map generator for on-the-fly map creation
sys.path.insert(0, str(Path(__file__).parent))
from generate_brainmaps import (
    generate_simulated_brainmap,
    generate_tribev2_brainmap,
    token_to_readable,
    token_signature,
    N_VERTICES,
)


# ── CORA class labels (for --mode cora) ──
CORA_CLASSES = [
    "Case_Based", "Genetic_Algorithms", "Neural_Networks",
    "Probabilistic_Methods", "Reinforcement_Learning",
    "Rule_Learning", "Theory",
]


# ───────────────────── Model Definitions ─────────────────────

class SignatureGCN(torch.nn.Module):
    """Multi-label GCN for signature detection (must match gnn_cora.py)."""

    def __init__(self, num_features, hidden1, hidden2, num_outputs, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(num_features, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.head = torch.nn.Linear(hidden2, num_outputs)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.head(x)
        return x


class CoraGCN(torch.nn.Module):
    """Standard 2-layer GCN (must match gnn_cora.py)."""

    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ──────────────── On-the-fly Brain Map Decoding ──────────────

def load_signature_model(model_path, graph_path, device):
    """Load trained signature GCN and graph data."""
    graph_data = torch.load(graph_path, map_location=device, weights_only=False)
    token_list = graph_data.token_list

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = SignatureGCN(
        num_features=ckpt["num_features"],
        hidden1=ckpt["hidden1"],
        hidden2=ckpt["hidden2"],
        num_outputs=ckpt["vocab_size"],
        dropout=ckpt.get("dropout", 0.5),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, graph_data, token_list


def compute_active_set(sig: np.ndarray, threshold: float = 0.1) -> set:
    """Get the set of active (above threshold) vertices in a signature."""
    return set(np.where(sig > threshold)[0])


def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if len(set_a) == 0 and len(set_b) == 0:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


@torch.no_grad()
def decode_query(model, graph_data, query: str, token_list: list[str],
                 threshold: float = 0.3, top_k: int = 15,
                 device: str = "cpu", activation_thresh: float = 0.1,
                 jaccard_thresh: float = 0.15):
    """Decode ANY query by generating its brain map and running through the GNN.

    Steps:
      1. Generate the query's brain map on-the-fly
      2. Add it as a temporary node to the graph
      3. Build edges to existing nodes via Jaccard overlap
      4. Run GNN forward pass on the augmented graph
      5. Read off which signatures the GNN detects as active
    """
    n_existing = graph_data.x.shape[0]

    # Step 1: Generate brain map for the query
    brain_map = generate_simulated_brainmap(query, n_timesteps=1)
    query_sig = brain_map.mean(axis=0)  # [N_VERTICES]
    query_tensor = torch.tensor(query_sig, dtype=torch.float32).unsqueeze(0)  # [1, V]

    # Step 2: Augment graph — add query as a new node
    augmented_x = torch.cat([graph_data.x.cpu(), query_tensor], dim=0)  # [N+1, V]

    # Step 3: Build edges between query node and existing nodes
    query_active = compute_active_set(query_sig, activation_thresh)
    new_sources, new_targets = [], []

    existing_sigs = graph_data.x.cpu().numpy()
    for i in range(n_existing):
        node_active = compute_active_set(existing_sigs[i], activation_thresh)
        jac = jaccard(query_active, node_active)
        if jac > jaccard_thresh:
            # Bidirectional edges
            new_sources.extend([n_existing, i])
            new_targets.extend([i, n_existing])

    # Combine existing edges with new edges
    if new_sources:
        new_edges = torch.tensor([new_sources, new_targets], dtype=torch.long)
        augmented_edges = torch.cat([graph_data.edge_index.cpu(), new_edges], dim=1)
    else:
        augmented_edges = graph_data.edge_index.cpu()

    # Step 4: Run GNN on augmented graph
    augmented_x = augmented_x.to(device)
    augmented_edges = augmented_edges.to(device)

    out = model(augmented_x, augmented_edges)  # [N+1, vocab_size]
    query_scores = torch.sigmoid(out[n_existing]).cpu().numpy()  # [vocab_size]

    # Step 5: Also compute direct cosine similarities for comparison
    query_norm = query_sig / (np.linalg.norm(query_sig) + 1e-8)
    cosine_scores = []
    for i in range(n_existing):
        node_sig = existing_sigs[i]
        node_norm = node_sig / (np.linalg.norm(node_sig) + 1e-8)
        cosine_scores.append(np.dot(query_norm, node_norm))
    cosine_scores = np.array(cosine_scores)

    # Combine GNN scores with cosine similarity (GNN captures graph propagation,
    # cosine captures direct signature overlap)
    combined_scores = 0.5 * query_scores[:n_existing] + 0.5 * cosine_scores

    # Sort and filter results
    sorted_indices = np.argsort(combined_scores)[::-1]

    results = []
    for idx in sorted_indices[:top_k]:
        score = float(combined_scores[idx])
        gnn_score = float(query_scores[idx])
        cos_score = float(cosine_scores[idx])
        if score < threshold and len(results) >= 3:
            break
        results.append({
            "token": token_list[idx],
            "score": score,
            "gnn_score": gnn_score,
            "cosine": cos_score,
        })

    return results


@torch.no_grad()
def decode_query_tribev2(model, graph_data, query: str, token_list: list[str],
                        tribe_model, duration: float = 30.0,
                        threshold: float = 0.3, top_k: int = 15,
                        device: str = "cpu", activation_thresh: float = 0.1,
                        jaccard_thresh: float = 0.15):
    """Decode ANY query using real TRIBE v2 brain maps.

    Same as decode_query but generates brain maps via TRIBE v2 instead of
    the simulated generator.
    """
    n_existing = graph_data.x.shape[0]

    # Step 1: Generate brain map via TRIBE v2
    readable = token_to_readable(query)
    brain_map = generate_tribev2_brainmap(
        tribe_model, query, readable, duration=duration)
    if brain_map is None or brain_map.shape[0] == 0:
        print(f"  ⚠ TRIBE v2 returned empty brain map for '{query}'")
        return []

    query_sig = brain_map.mean(axis=0)  # [N_VERTICES]
    query_tensor = torch.tensor(query_sig, dtype=torch.float32).unsqueeze(0)  # [1, V]

    # Step 2: Augment graph — add query as a new node
    augmented_x = torch.cat([graph_data.x.cpu(), query_tensor], dim=0)  # [N+1, V]

    # Step 3: Build edges between query node and existing nodes
    query_active = compute_active_set(query_sig, activation_thresh)
    new_sources, new_targets = [], []

    existing_sigs = graph_data.x.cpu().numpy()
    for i in range(n_existing):
        node_active = compute_active_set(existing_sigs[i], activation_thresh)
        jac = jaccard(query_active, node_active)
        if jac > jaccard_thresh:
            new_sources.extend([n_existing, i])
            new_targets.extend([i, n_existing])

    if new_sources:
        new_edges = torch.tensor([new_sources, new_targets], dtype=torch.long)
        augmented_edges = torch.cat([graph_data.edge_index.cpu(), new_edges], dim=1)
    else:
        augmented_edges = graph_data.edge_index.cpu()

    # Step 4: Run GNN on augmented graph
    augmented_x = augmented_x.to(device)
    augmented_edges = augmented_edges.to(device)

    out = model(augmented_x, augmented_edges)
    query_scores = torch.sigmoid(out[n_existing]).cpu().numpy()

    # Step 5: Cosine similarities
    query_norm = query_sig / (np.linalg.norm(query_sig) + 1e-8)
    cosine_scores = []
    for i in range(n_existing):
        node_sig = existing_sigs[i]
        node_norm = node_sig / (np.linalg.norm(node_sig) + 1e-8)
        cosine_scores.append(np.dot(query_norm, node_norm))
    cosine_scores = np.array(cosine_scores)

    combined_scores = 0.5 * query_scores[:n_existing] + 0.5 * cosine_scores
    sorted_indices = np.argsort(combined_scores)[::-1]

    results = []
    for idx in sorted_indices[:top_k]:
        score = float(combined_scores[idx])
        gnn_score = float(query_scores[idx])
        cos_score = float(cosine_scores[idx])
        if score < threshold and len(results) >= 3:
            break
        results.append({
            "token": token_list[idx],
            "score": score,
            "gnn_score": gnn_score,
            "cosine": cos_score,
        })

    return results


def print_decode_result(query: str, results: list[dict]):
    """Pretty-print decoded token signatures."""
    print(f"\n  Input: \"{query}\"")
    print(f"  {'─'*60}")
    print(f"  Active signatures detected:")
    print()
    print(f"    {'Token':17s}  {'Score':>6s}  {'GNN':>5s}  {'Cos':>5s}  {'Bar'}")
    print(f"    {'─'*55}")

    for r in results:
        score = r["score"]
        bar_len = int(score * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)

        if score > 0.7:
            label = "← STRONG"
        elif score > 0.4:
            label = "← active"
        elif score > 0.2:
            label = "← weak"
        else:
            label = ""

        print(f"    \"{r['token']:15s}\"  {score:.4f}  {r['gnn_score']:.3f}  "
              f"{r['cosine']:.3f}  {bar} {label}")

    # Reconstruct decoded thought from strong signals
    strong = [r["token"] for r in results if r["score"] > 0.4]
    if strong:
        print(f"\n  💭 Decoded thought: \"{' '.join(strong)}\"")
    print()


def interactive_mode(model, graph_data, token_list, threshold, device,
                     tribe_model=None, duration=30.0):
    """Interactive REPL for querying the brain decoder."""
    use_tribev2 = tribe_model is not None
    mode_str = "TRIBE v2" if use_tribev2 else "Simulated"
    print(f"\n{'='*60}")
    print(f"  Brain Signature Decoder — Interactive Mode")
    print(f"{'='*60}")
    print(f"  Mode:            {mode_str}")
    print(f"  Vocabulary size: {len(token_list)}")
    print(f"  Threshold:       {threshold}")
    if use_tribev2:
        print(f"  Duration:        {duration}s per query")
    print(f"  Type ANY expression or word to decode.")
    print(f"  Examples: 1+1, 5+3, dog, two, 10-7, 4*5")
    print(f"  Commands: 'list' = show vocab, 'quit' = exit")
    print(f"{'='*60}\n")

    while True:
        try:
            query = input("  🧠 Enter query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break
        if query.lower() == "list":
            print(f"\n  Vocabulary ({len(token_list)} tokens):")
            for i, t in enumerate(token_list):
                print(f"    [{i:4d}] {t}")
            print()
            continue
        if query.lower() == "help":
            print("  Enter any token or expression.")
            print("  The decoder generates a brain map, runs it through")
            print("  the GNN, and shows which signatures are active.")
            print("  Examples: 1+1, 5*3, dog, twenty, 15-8")
            continue

        if use_tribev2:
            results = decode_query_tribev2(
                model, graph_data, query, token_list,
                tribe_model=tribe_model, duration=duration,
                threshold=threshold, device=device,
            )
        else:
            results = decode_query(
                model, graph_data, query, token_list,
                threshold=threshold, device=device,
            )
        print_decode_result(query, results)


def batch_evaluate(model, graph_data, token_list, threshold, device,
                   tribe_model=None, duration=30.0):
    """Evaluate on some test expressions."""
    use_tribev2 = tribe_model is not None
    print(f"\n{'='*60}")
    print(f"Batch Evaluation ({'TRIBE v2' if use_tribev2 else 'Simulated'})")
    print(f"{'='*60}")

    test_queries = [
        "1+1", "2+3", "5+5", "10-7", "3*3", "4*5",
        "1", "2", "dog", "five", "twenty",
    ]

    for query in test_queries:
        if use_tribev2:
            results = decode_query_tribev2(
                model, graph_data, query, token_list,
                tribe_model=tribe_model, duration=duration,
                threshold=threshold, device=device,
            )
        else:
            results = decode_query(
                model, graph_data, query, token_list,
                threshold=threshold, device=device,
            )
        print_decode_result(query, results)


# ──────────────── CORA Prediction ──────────────────

def cora_predict(model_path, data_dir, node_idx, device):
    """Standard CORA prediction."""
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures

    dataset = Planetoid(root=data_dir, name="Cora", transform=NormalizeFeatures())
    data = dataset[0]

    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model = CoraGCN(
        num_features=ckpt["num_features"],
        hidden_channels=ckpt["hidden_channels"],
        num_classes=ckpt["num_classes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        probs = F.softmax(out, dim=1).cpu()
        pred = probs.argmax(dim=1)

    if node_idx is not None:
        p = probs[node_idx]
        print(f"\n  Node {node_idx}: pred={CORA_CLASSES[p.argmax()]}, true={CORA_CLASSES[data.y[node_idx]]}")
        for i, cls in enumerate(CORA_CLASSES):
            print(f"    {cls:25s}  {p[i].item():.4f}  {'█' * int(p[i].item() * 30)}")
    else:
        for split, mask in [("Train", data.train_mask), ("Val", data.val_mask),
                            ("Test", data.test_mask)]:
            correct = (pred[mask] == data.y[mask]).sum().item()
            total = mask.sum().item()
            print(f"  {split:6s} accuracy: {correct / total:.4f} ({correct}/{total})")


# ─────────────────────── Main ────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Brain Signature Decoder")
    parser.add_argument("--mode", type=str, default="signature",
                        choices=["signature", "cora"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--graph", type=str, default="signature_graph.pt")
    parser.add_argument("--word", type=str, default=None,
                        help="Decode a specific word/expression")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--node", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default="data/Cora")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Event duration for TRIBE v2 brain map generation")
    parser.add_argument("--use-tribev2", action="store_true",
                        help="Use real TRIBE v2 for on-the-fly brain maps")
    parser.add_argument("--tribev2-checkpoint", type=str, default="facebook/tribev2")
    parser.add_argument("--tribev2-device", type=str, default="cpu",
                        help="Device for TRIBE v2 model")
    args = parser.parse_args()

    if args.model is None:
        args.model = "model_signature.pt" if args.mode == "signature" else "model_cora.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "signature":
        if not Path(args.model).exists():
            print(f"Error: {args.model} not found. Run 'python gnn_cora.py' first.")
            sys.exit(1)
        if not Path(args.graph).exists():
            print(f"Error: {args.graph} not found. Run 'python build_signature_graph.py' first.")
            sys.exit(1)

        model, graph_data, token_list = load_signature_model(
            args.model, args.graph, device)
        graph_data = graph_data.to(device)

        # Load TRIBE v2 if requested
        tribe_model = None
        if args.use_tribev2:
            import os
            if args.tribev2_device == "cpu":
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
            sys.path.insert(0, str(Path(__file__).parent / "tribev2"))
            from tribev2 import TribeModel
            print(f"\n  ⏳ Loading TRIBE v2 for inference...")
            tribe_model = TribeModel.from_pretrained(
                args.tribev2_checkpoint, cache_folder="./cache",
                device=args.tribev2_device)
            print(f"  ✓ TRIBE v2 loaded!\n")

        if args.evaluate:
            batch_evaluate(model, graph_data, token_list, args.threshold, device,
                          tribe_model=tribe_model, duration=args.duration)
        elif args.word:
            if tribe_model:
                results = decode_query_tribev2(
                    model, graph_data, args.word, token_list,
                    tribe_model=tribe_model, duration=args.duration,
                    threshold=args.threshold, device=device,
                )
            else:
                results = decode_query(
                    model, graph_data, args.word, token_list,
                    threshold=args.threshold, device=device,
                )
            print_decode_result(args.word, results)
        else:
            interactive_mode(model, graph_data, token_list, args.threshold, device,
                           tribe_model=tribe_model, duration=args.duration)

    else:
        if not Path(args.model).exists():
            print(f"Error: {args.model} not found. Run 'python gnn_cora.py --mode cora' first.")
            sys.exit(1)
        cora_predict(args.model, args.data_dir, args.node, device)


if __name__ == "__main__":
    main()
