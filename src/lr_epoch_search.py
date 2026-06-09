"""
Search best learning rate and epoch count for hidden sizes 1024 and 512.

Usage:
PYTHONPATH=src python3 src/lr_epoch_search.py
PYTHONPATH=src python3 src/lr_epoch_search.py --learning-rates 0.1 0.05 0.01 --epochs 1 5 10
"""
from __future__ import annotations

import argparse
import csv
import time
from typing import List

from load_fashion_mnist import load_train_data
from network import NetworkConfig, SimpleMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search learning rate and epochs")
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=[0.1, 0.05, 0.01],
        help="Learning rates to try",
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=[30,60,90],
        help="Epoch counts to try",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="lr_epoch_search_results.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    (x_train, t_train), (x_valid, t_valid) = load_train_data()

    results = []

    for lr in args.learning_rates:
        for epochs in args.epochs:
            print(f"Testing lr={lr}, epochs={epochs} ...")
            cfg = NetworkConfig(
                input_size=x_train.shape[1],
                hidden_size=1024,
                hidden_size2=512,
                output_size=10,
                learning_rate=lr,
                batch_size=args.batch_size,
                seed=42,
            )
            model = SimpleMLP(cfg)

            start = time.perf_counter()
            loss = 0.0
            for epoch in range(epochs):
                loss = model.train_epoch(x_train, t_train, epoch=epoch)
            elapsed = time.perf_counter() - start

            train_acc = model.evaluate_accuracy(x_train, t_train)
            valid_acc = model.evaluate_accuracy(x_valid, t_valid)

            print(
                f"lr={lr}, epochs={epochs} -> loss={loss:.4f}, "
                f"train={train_acc:.4f}, valid={valid_acc:.4f}, time={elapsed:.2f}s"
            )

            results.append(
                {
                    "learning_rate": lr,
                    "epochs": epochs,
                    "loss": float(loss),
                    "train_acc": float(train_acc),
                    "valid_acc": float(valid_acc),
                    "time_s": float(elapsed),
                }
            )

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "learning_rate",
                "epochs",
                "loss",
                "train_acc",
                "valid_acc",
                "time_s",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    best = max(results, key=lambda row: (row["valid_acc"], -row["loss"]))
    print(
        "Best result:",
        f"lr={best['learning_rate']} epochs={best['epochs']} "
        f"valid_acc={best['valid_acc']:.4f} train_acc={best['train_acc']:.4f}",
    )
    print(f"Saved results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())