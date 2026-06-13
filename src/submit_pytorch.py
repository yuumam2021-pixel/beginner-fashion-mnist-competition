import torch
from pathlib import Path
from load_fashion_mnist import load_test_data
from train_pytorch import SuperResNet

def main() -> int:
    NUM_MODELS = 3
    
    # 1. テストデータの読み込み
    x_test_np, t_test_np = load_test_data()
    x_test = torch.from_numpy(x_test_np).view(-1, 1, 28, 28).float()
    t_test = torch.from_numpy(t_test_np).long()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_test = x_test.to(device)
    t_test = t_test.to(device)

    # 2. 3つのモデルをすべて準備して読み込む
    models = []
    for i in range(NUM_MODELS):
        weight_path = Path(f"sample_weight_{i}.pkl")
        if not weight_path.exists():
            print(f"[ERROR] weights file not found: {weight_path}")
            return 1
            
        model = SuperResNet()
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)
        print(f"✅ モデル {i+1} ({weight_path}) を読み込みました。")

    # 3. アンサンブル予測を実行
    print("🔥 3つのモデルによるアンサンブル予測を計算中...")
    
    with torch.no_grad():
        # 全モデルの予測結果（確率）を足し合わせるための空っぽのテンソル
        ensemble_outputs = torch.zeros((x_test.size(0), 10)).to(device)
        
        for model in models:
            # 各モデルでオリジナル画像 ＋ 左右反転画像（TTA）の予測を出す
            out_orig = model(x_test)
            out_flip = model(torch.flip(x_test, dims=[3]))
            model_output = (out_orig + out_flip) / 2.0
            
            # アンサンブル用に足し合わせる
            ensemble_outputs += model_output
            
        # 足し合わせたものをモデルの数(3)で割って平均を出す
        ensemble_outputs /= NUM_MODELS
        
        # 最終的な答えを決める
        _, predicted = torch.max(ensemble_outputs, 1)
        
        # 精度の計算
        correct = (predicted == t_test).sum().item()
        total = t_test.size(0)

    test_acc = correct / total
    print(f"\n👑 【最終結果】 アンサンブル テスト正解率 (Test Acc): {test_acc:.4f} 👑")
    
    return 0

if __name__ == "__main__":
    main()