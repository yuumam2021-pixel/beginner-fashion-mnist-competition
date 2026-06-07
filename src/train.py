# uv run src/train.py

import pickle
from pathlib import Path

from load_fashion_mnist import load_train_data
from network import NetworkConfig, SimpleMLP

OUTPUT_PATH = Path("sample_weight.pkl")
EPOCHS = 60
HIDDEN_SIZE = 512
LEARNING_RATE = 0.05
BATCH_SIZE = 128
SEED = 42


def main() -> int:
    (x_train, t_train), (x_valid, t_valid) = load_train_data()

    model = SimpleMLP(
        NetworkConfig(
            input_size=x_train.shape[1],
            hidden_size=HIDDEN_SIZE,
            output_size=10,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            seed=SEED,
        )
    )

    # 1. 設定の準備（先ほどのアドバイス通り hidden_size を 256 に増やしてみます）
    max_epochs = EPOCHS

    print("学習をスタートします！")

    # 2. ここでエポック数（学習回数）を決めます！例えば 30 回に増やしてみます
    for epoch in range(max_epochs):
        # 3. 指定した回数だけ学習を繰り返すループ
        loss = model.train_epoch(x_train, t_train, epoch=epoch)
        train_acc = model.evaluate_accuracy(x_train, t_train)
        valid_acc = model.evaluate_accuracy(x_valid, t_valid)
        print(
            f"Epoch {epoch + 1:02d} | Loss: {loss:.4f} | "
            f"Train Accuracy: {train_acc:.4f} | Valid Accuracy: {valid_acc:.4f}"
        )

    print("学習が完了しました！")

    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(model.to_state(), f)

    print(f"Saved model: {OUTPUT_PATH.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
