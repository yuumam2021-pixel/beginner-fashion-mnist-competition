import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from load_fashion_mnist import load_train_data

# ==========================================
# 1. 限界突破：超強力3層CNNモデル（Dropout搭載）
# ==========================================
class DeepCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1層目: 1 -> 32チャンネル
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # 2層目: 32 -> 64チャンネル
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 💡 3層目: 64 -> 128チャンネル（さらに高度な特徴を抽出）
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 💡 3回のプーリングで画像は 3x3 マスまで縮小されます
        # 128チャンネル × 3マス × 3マス = 1152
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        
        # 💡 秘密兵器：Dropout（25%の確率でランダムにニューロンを休ませ、丸暗記を防ぐ）
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = self.pool(F.relu(self.conv3(x)))  # 7x7 -> 3x3
        
        # 1列に平坦化
        x = x.view(-1, 128 * 3 * 3)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 💡 全結合層の後にドロップアウトを適用！
        x = self.fc2(x)
        return x

# ==========================================
# 2. 学習メイン処理
# ==========================================
def main():
    # 1. データの読み込み
    (x_train_np, t_train_np), (x_valid_np, t_valid_np) = load_train_data()

    # NumPyデータをPyTorchテンソルに変換 ＆ 2D (28x28) に変形
    x_train = torch.from_numpy(x_train_np).view(-1, 1, 28, 28).float()
    t_train = torch.from_numpy(t_train_np).long()
    x_valid = torch.from_numpy(x_valid_np).view(-1, 1, 28, 28).float()
    t_valid = torch.from_numpy(t_valid_np).long()

    # ミニバッチの自動化
    train_dataset = TensorDataset(x_train, t_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # モデル、損失関数、最適化の準備
    model = DeepCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 💡 学習率スケジューラー（10エポックごとに学習率を半分にする）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epochs = 30  # より深く賢くなるので、30エポックじっくり回します
    print("限界突破：3層CNNの学習をスタートします！")

    for epoch in range(epochs):
        model.train()  # 💡学習モード（Dropoutが有効になります）
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # エポック終了時に学習率を更新
        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train

        # 検証データでの精度評価
        model.eval()  # 💡評価モード（Dropoutが無効になり、100%の実力を出します）
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            outputs_valid = model(x_valid)
            _, predicted_valid = torch.max(outputs_valid, 1)
            total_valid += t_valid.size(0)
            correct_valid += (predicted_valid == t_valid).sum().item()
        
        valid_acc = correct_valid / total_valid

        print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f} | Valid Acc: {valid_acc:.4f}")

    print("学習が完了しました！")
    # 💡 新しい最強モデルを保存
    torch.save(model.state_dict(), "pytorch_model.pth")
    print("モデルを pytorch_model.pth に保存しました！")

if __name__ == "__main__":
    main()