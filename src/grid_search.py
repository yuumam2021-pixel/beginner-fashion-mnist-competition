"""
Grid search for hidden layer sizes (runs short experiments and saves results).

Usage (from project root):

PYTHONPATH=src python3 src/grid_search.py --epochs 1

Or with virtualenv:
PYTHONPATH=src .venv/bin/python src/grid_search.py --epochs 1
"""

from __future__ import annotations

import argparse
import csv
import time
from typing import List, Tuple

import numpy as np

from load_fashion_mnist import load_train_data
from network import NetworkConfig, SimpleMLP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search for hidden layer sizes")
    p.add_argument(
        "--grid",
        nargs="*",
        default=None,
        help=(
            "List of hidden-size pairs as H1,H2 (e.g. 1024,512). "
            "If omitted uses default preset."
        ),
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--output", default="grid_search_results.csv")
    return p.parse_args()

def parse_grid(items: List[str]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for it in items:
        try:
            a, b = it.split(",")
            out.append((int(a), int(b)))
        except Exception:
            raise ValueError(f"Invalid grid item: {it}. Expected format H1,H2")
    return out

def main() -> int:
    args = parse_args()

    if args.grid:
        grid = parse_grid(args.grid)
    else:
        grid = [(2048, 1024), (1024, 512), (512, 256), (256, 128), (128, 64)]

    (x_train, t_train), (x_valid, t_valid) = load_train_data()

    results = []

    for h1, h2 in grid:
        print(f"Running h1={h1}, h2={h2} (epochs={args.epochs})...")
        cfg = NetworkConfig(
            input_size=x_train.shape[1],
            hidden_size=h1,
            hidden_size2=h2,
            output_size=10,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            seed=42,
        )
        model = SimpleMLP(cfg)

        start = time.perf_counter()
        for e in range(args.epochs):
            loss = model.train_epoch(x_train, t_train, epoch=e)
        elapsed = time.perf_counter() - start

        train_acc = model.evaluate_accuracy(x_train, t_train)
        valid_acc = model.evaluate_accuracy(x_valid, t_valid)

        print(
            f"Result h1={h1}, h2={h2} -> Loss:{loss:.4f}, Train:{train_acc:.4f}, Valid:{valid_acc:.4f}, Time:{elapsed:.2f}s"
        )

        results.append(
            {
                "h1": h1,
                "h2": h2,
                "loss": float(loss),
                "train_acc": float(train_acc),
                "valid_acc": float(valid_acc),
                "time_s": float(elapsed),
            }
        )

    # save CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["h1", "h2", "loss", "train_acc", "valid_acc", "time_s"]
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Saved results to {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())