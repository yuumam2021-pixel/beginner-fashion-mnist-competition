import torch
from pathlib import Path
from load_fashion_mnist import load_test_data
# 💡 ここを SuperResNet に修正！
from train_pytorch import SuperResNet as DeepCNN

WEIGHTS_PATH = Path("sample_weight.pkl")

def main() -> int:
    if not WEIGHTS_PATH.exists():
        print(f"[ERROR] weights file not found: {WEIGHTS_PATH}")
        return 1

    # # 1. テストデータの読み込み
    x_test_np, t_test_np = load_test_data()

    # # 2. NumPyデータをPyTorchテンソルに変換 & 28x28に変形
    x_test = torch.from_numpy(x_test_np).view(-1, 1, 28, 28).float()
    t_test = torch.from_numpy(t_test_np).long()

    # # 3. モデルを準備して重みを流し込む
    model = DeepCNN()
    # CPU環境でも読み込めるように map_location を追加しておくと安全です
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval() # 💡 テストモードに切り替え（重要！）

    # # 4. 予測を実行
    print("テストデータで予測を計算中...")
    correct = 0
    total = 0
    
    with torch.no_grad():
        # GPUが使えるならデータをGPUに送る
        images = x_test.to(device)
        labels = t_test.to(device)
        
        # 予測（TTA: 左右反転も合わせてアンサンブルするとさらに精度が上がります）
        outputs_orig = model(images)
        outputs_flipped = model(torch.flip(images, dims=[3]))
        outputs = (outputs_orig + outputs_flipped) / 2.0
        
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # # 5. 精度の計算
    test_acc = correct / total
    print(f"✨ テストデータの正解率 (Test Acc): {test_acc:.4f}")
    
    # ここで提出用のCSV等を作る処理があれば、ここ以降に続きます
    return 0

if __name__ == "__main__":
    main()