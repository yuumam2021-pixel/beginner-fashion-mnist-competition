# src/submit_pytorch.py

import torch
from pathlib import Path
from load_fashion_mnist import load_test_data
from train_pytorch import DeepCNN  # 💡 最強になった DeepCNN を読み込む

WEIGHTS_PATH = Path("sample_weight.pkl")
model = DeepCNN()
model.load_state_dict(torch.load(WEIGHTS_PATH))

def main() -> int:
    if not WEIGHTS_PATH.exists():
        print(f"[ERROR] weights file not found: {WEIGHTS_PATH}")
        return 1

    # 1. テストデータの読み込み
    x_test_np, t_test_np = load_test_data()

    # 2. NumPyデータをPyTorchテンソルに変換 ＆ 28x28に変形
    x_test = torch.from_numpy(x_test_np).view(-1, 1, 28, 28).float()
    t_test = torch.from_numpy(t_test_np).long()

    # 3. モデルを準備して重みを流し込む
    model = DeepCNN()
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    model.eval()  # 💡テストモードに切り替え（重要！）

    # 4. 予測を実行
    with torch.no_grad():
        outputs = model(x_test)
        _, predicted = torch.max(outputs, 1)

    # 5. 精度の計算
    correct = (predicted == t_test).sum().item()
    total = t_test.size(0)
    acc = correct / total

    print(f"★最終結果★ Test Accuracy: {acc:.6f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())