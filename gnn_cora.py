"""
gnn_cora.py — GCN training on the Brain Signature Graph

Architecture follows README_BRAIN_DECODER:
  - 2-layer GCN: [N_VERTICES] → [256] → [128]
  - Linear head: [128] → [vocab_size]
  - Sigmoid output (multi-label, not softmax)
  - BCE loss against Jaccard overlap targets

Can also run on standard CORA dataset with --mode cora flag.

Usage:
    python gnn_cora.py                                # train on signature graph
    python gnn_cora.py --mode cora                    # train on standard CORA
    python gnn_cora.py --graph signature_graph.pt     # custom graph path
    python gnn_cora.py --epochs 300 --lr 0.005        # custom hyperparams
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


# ───────────────────── Signature GCN Model ───────────────────

class SignatureGCN(nn.Module):
    """GCN for multi-label signature detection.

    Architecture (from README):
        input   [N, N_VERTICES]
            ↓   Dropout(0.5)
        GCNConv [N, N_VERTICES] → [N, 256]
            ↓   ReLU + Dropout(0.5)
        GCNConv [N, 256] → [N, 128]
            ↓   ReLU
        Linear  [N, 128] → [N, vocab_size]
            ↓   Sigmoid (multi-label)
    """

    def __init__(self, num_features: int, hidden1: int, hidden2: int,
                 num_outputs: int, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(num_features, hidden1)
        self.conv2 = GCNConv(hidden1, hidden2)
        self.head = nn.Linear(hidden2, num_outputs)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.head(x)
        return x  # raw logits — sigmoid applied in loss


# ──────────────────── Standard CORA GCN ──────────────────────

class CoraGCN(nn.Module):
    """Standard 2-layer GCN for CORA node classification (7 classes)."""

    def __init__(self, num_features: int, hidden_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ─────────────── Signature Training Functions ────────────────

def train_signature(model, data, optimizer):
    """Train step for multi-label signature detection using BCE."""
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)  # [N, vocab_size]
    preds = torch.sigmoid(out)

    # Target: Jaccard overlap matrix (soft multi-label targets)
    targets = data.jaccard  # [N, N] — jaccard[i][j] = overlap of sig_i and sig_j

    # Loss only on training nodes
    mask = data.train_mask
    loss = F.binary_cross_entropy(preds[mask], targets[mask])

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_signature(model, data):
    """Evaluate signature detection accuracy.

    For each token, check if the top-K predicted signatures match
    the tokens with highest Jaccard overlap.
    """
    model.eval()
    out = model(data.x, data.edge_index)
    preds = torch.sigmoid(out)
    targets = data.jaccard

    results = {}
    for split, mask in [("train", data.train_mask),
                        ("val", data.val_mask),
                        ("test", data.test_mask)]:
        if mask.sum() == 0:
            continue

        # Self-identification accuracy: does each node score itself highest?
        masked_preds = preds[mask]
        masked_indices = torch.where(mask)[0]

        self_scores = []
        for i, node_idx in enumerate(masked_indices):
            self_score = masked_preds[i, node_idx].item()
            self_scores.append(self_score)
        avg_self_score = sum(self_scores) / len(self_scores)

        # Top-K overlap accuracy
        k = 5
        top_k_correct = 0
        total = 0
        for i, node_idx in enumerate(masked_indices):
            pred_topk = masked_preds[i].topk(k).indices.tolist()
            true_topk = targets[node_idx].topk(k).indices.tolist()
            overlap = len(set(pred_topk) & set(true_topk))
            top_k_correct += overlap
            total += k
        topk_acc = top_k_correct / total if total > 0 else 0.0

        # BCE loss
        loss = F.binary_cross_entropy(preds[mask], targets[mask]).item()

        results[split] = {
            "loss": loss,
            "self_score": avg_self_score,
            "topk_acc": topk_acc,
        }

    return results


# ──────────────── Standard CORA Training ─────────────────────

def train_cora(model, data, optimizer, criterion):
    """Standard CORA training step."""
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_cora(model, data):
    """Standard CORA evaluation."""
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = {}
    for split, mask in [("train", data.train_mask),
                        ("val", data.val_mask),
                        ("test", data.test_mask)]:
        correct = pred[mask] == data.y[mask]
        accs[split] = int(correct.sum()) / int(mask.sum())
    return accs


# ─────────────────────── Main entry ──────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train GCN on signature graph or CORA")
    parser.add_argument("--mode", type=str, default="signature",
                        choices=["signature", "cora"],
                        help="Training mode: signature graph or standard CORA")
    parser.add_argument("--graph", type=str, default="signature_graph.pt",
                        help="Path to signature graph (signature mode)")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--wd", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--hidden1", type=int, default=256, help="First hidden dim")
    parser.add_argument("--hidden2", type=int, default=128, help="Second hidden dim")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", type=str, default=None,
                        help="Model save path (default: auto)")
    parser.add_argument("--data-dir", type=str, default="data/Cora",
                        help="CORA dataset directory (cora mode only)")
    args = parser.parse_args()

    if args.save is None:
        args.save = "model_signature.pt" if args.mode == "signature" else "model_cora.pt"

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode:   {args.mode}")

    # ════════════════════ SIGNATURE MODE ════════════════════
    if args.mode == "signature":
        # Load signature graph
        data = torch.load(args.graph, map_location=device, weights_only=False)
        token_list = data.token_list
        vocab_size = len(token_list)

        print(f"\n{'='*60}")
        print(f"Signature Graph")
        print(f"{'='*60}")
        print(f"  Tokens:           {vocab_size}")
        print(f"  Edges:            {data.num_edges}")
        print(f"  Features/token:   {data.x.shape[1]}")
        print(f"  Train tokens:     {int(data.train_mask.sum())}")
        print(f"  Val tokens:       {int(data.val_mask.sum())}")
        print(f"  Test tokens:      {int(data.test_mask.sum())}")
        print(f"{'='*60}\n")

        # Move data to device
        data = data.to(device)

        # Build model
        model = SignatureGCN(
            num_features=data.x.shape[1],
            hidden1=args.hidden1,
            hidden2=args.hidden2,
            num_outputs=vocab_size,
            dropout=args.dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}\n")

        # Training loop
        best_val_loss = float("inf")
        best_epoch = 0

        for epoch in range(1, args.epochs + 1):
            loss = train_signature(model, data, optimizer)
            results = evaluate_signature(model, data)

            val_loss = results.get("val", {}).get("loss", float("inf"))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "num_features": data.x.shape[1],
                    "hidden1": args.hidden1,
                    "hidden2": args.hidden2,
                    "vocab_size": vocab_size,
                    "dropout": args.dropout,
                    "token_list": token_list,
                    "val_loss": best_val_loss,
                }, args.save)

            if epoch % 10 == 0 or epoch == 1:
                train_r = results.get("train", {})
                val_r = results.get("val", {})
                test_r = results.get("test", {})
                print(f"Epoch {epoch:03d}  |  Loss: {loss:.4f}  |  "
                      f"Self: {train_r.get('self_score', 0):.3f}/{val_r.get('self_score', 0):.3f}/{test_r.get('self_score', 0):.3f}  |  "
                      f"Top5: {train_r.get('topk_acc', 0):.3f}/{val_r.get('topk_acc', 0):.3f}/{test_r.get('topk_acc', 0):.3f}")

        # Final results
        print(f"\n{'='*60}")
        print(f"Training complete")
        print(f"{'='*60}")
        print(f"  Best val loss:  {best_val_loss:.4f} (epoch {best_epoch})")

        # Reload best
        ckpt = torch.load(args.save, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        final = evaluate_signature(model, data)
        for split in ["train", "val", "test"]:
            r = final.get(split, {})
            print(f"  {split:6s}  loss={r.get('loss', 0):.4f}  "
                  f"self_score={r.get('self_score', 0):.4f}  "
                  f"top5_acc={r.get('topk_acc', 0):.4f}")
        print(f"  Model saved to: {args.save}")
        print(f"{'='*60}")

    # ════════════════════ CORA MODE ════════════════════
    else:
        dataset = Planetoid(root=args.data_dir, name="Cora", transform=NormalizeFeatures())
        data = dataset[0].to(device)

        print(f"\n{'='*60}")
        print(f"CORA Dataset")
        print(f"{'='*60}")
        print(f"  Nodes:          {data.num_nodes}")
        print(f"  Edges:          {data.num_edges}")
        print(f"  Features/node:  {dataset.num_features}")
        print(f"  Classes:        {dataset.num_classes}")
        print(f"{'='*60}\n")

        model = CoraGCN(
            num_features=dataset.num_features,
            hidden_channels=16,
            num_classes=dataset.num_classes,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        criterion = torch.nn.CrossEntropyLoss()

        print(model)
        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(1, args.epochs + 1):
            loss = train_cora(model, data, optimizer, criterion)
            accs = evaluate_cora(model, data)

            if accs["val"] > best_val_acc:
                best_val_acc = accs["val"]
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "num_features": dataset.num_features,
                    "hidden_channels": 16,
                    "num_classes": dataset.num_classes,
                    "val_acc": best_val_acc,
                }, args.save)

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d}  |  Loss: {loss:.4f}  |  "
                      f"Train: {accs['train']:.4f}  Val: {accs['val']:.4f}  "
                      f"Test: {accs['test']:.4f}")

        print(f"\n{'='*60}")
        print(f"  Best val acc: {best_val_acc:.4f} (epoch {best_epoch})")
        print(f"  Model saved to: {args.save}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
