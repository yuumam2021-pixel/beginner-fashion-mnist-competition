import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from load_fashion_mnist import load_train_data

# ==========================================
# 1. GoogLeNetの心臓部：Inceptionモジュール
# ==========================================
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super().__init__()
        
        # ルート1：1x1 畳み込み
        self.branch1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # ルート2：1x1 で次元削減してから 3x3 畳み込み
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        # ルート3：1x1 で次元削減してから 5x5 畳み込み
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        # ルート4：3x3 MaxPool してから 1x1 畳み込み
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1)
        )

    def forward(self, x):
        out1 = F.relu(self.branch1(x))
        out2 = F.relu(self.branch2(x))
        out3 = F.relu(self.branch3(x))
        out4 = F.relu(self.branch4(x))
        # 4つの並列ルートの出力をチャンネル方向に結合
        return torch.cat([out1, out2, out3, out4], dim=1)


# ==========================================
# 2. Inceptionを組み込んだ Mini-GoogLeNet
# ==========================================
class MiniGoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 最初の下処理：1 -> 32チャンネルへ
        self.prepare = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 💡 Inceptionモジュール1個目（出力は計64チャンネル）
        self.inception1 = InceptionModule(
            in_channels=32, 
            out_1x1=16, red_3x3=16, out_3x3=16, red_5x5=8, out_5x5=16, out_pool=16
        )
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 28x28 -> 14x14
        
        # 💡 Inceptionモジュール2個目（出力は計128チャンネル）
        self.inception2 = InceptionModule(
            in_channels=64, 
            out_1x1=32, red_3x3=32, out_3x3=32, red_5x5=16, out_5x5=32, out_pool=32
        )
        
        # もう一度プーリング：14x14 -> 7x7
        
        # 全結合層（128チャンネル × 7マス × 7マス = 6272）
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.prepare(x)        # 28x28
        x = self.inception1(x)     # 28x28
        x = self.pool(x)           # 14x14
        
        x = self.inception2(x)     # 14x14
        x = self.pool(x)           # 7x7
        
        x = x.view(-1, 128 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==========================================
# 3. 学習メイン処理
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

    # 💡 モデルに MiniGoogLeNet を指定！
    model = MiniGoogLeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 学習率スケジューラー（10エポックごとに学習率を半分にする）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epochs = 30
    print("限界突破：Mini-GoogLeNet（Inception）の学習をスタートします！")

    for epoch in range(epochs):
        model.train()  # 学習モード
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
        model.eval()  # 評価モード
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
    # 💡 指定通り sample_weight.pkl という名前で保存します
    torch.save(model.state_dict(), "sample_weight.pkl")
    print("モデルを sample_weight.pkl に保存しました！")


if __name__ == "__main__":
    main()