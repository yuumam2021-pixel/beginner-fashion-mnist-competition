import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from load_fashion_mnist import load_train_data

# ==========================================
# 1. 兵器①搭載：Batch Normalization版 Inceptionモジュール
# ==========================================
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super().__init__()
        
        # ルート1：1x1 畳み込み + バッチ正規化
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm2d(out_1x1), # 💡これでお掃除
            nn.ReLU()
        )
        
        # ルート2：1x1で次元削減 -> 3x3畳み込み + バッチ正規化
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.BatchNorm2d(red_3x3),
            nn.ReLU(),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_3x3), # 💡これでお掃除
            nn.ReLU()
        )
        
        # ルート3：1x1で次元削減 -> 5x5畳み込み + バッチ正規化
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.BatchNorm2d(red_5x5),
            nn.ReLU(),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_5x5), # 💡これでお掃除
            nn.ReLU()
        )
        
        # ルート4：3x3 MaxPool -> 1x1畳み込み + バッチ正規化
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1),
            nn.BatchNorm2d(out_pool), # 💡これでお掃除
            nn.ReLU()
        )

    def forward(self, x):
        # 各ルートを通ったあとにチャンネル結合
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


# ==========================================
# 2. 究極進化版：Mini-GoogLeNet
# ==========================================
class MiniGoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 最初の下処理にもバッチ正規化を搭載
        self.prepare = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Inceptionモジュール1個目（出力は計64チャンネル）
        self.inception1 = InceptionModule(
            in_channels=32, 
            out_1x1=16, red_3x3=16, out_3x3=16, red_5x5=8, out_5x5=16, out_pool=16
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 28x28 -> 14x14
        
        # Inceptionモジュール2個目（出力は計128チャンネル）
        self.inception2 = InceptionModule(
            in_channels=64, 
            out_1x1=32, red_3x3=32, out_3x3=32, red_5x5=16, out_5x5=32, out_pool=32
        )
        
        # 全結合層（128チャンネル × 7マス × 7マス = 6272）
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.bn_fc1 = nn.BatchNorm1d(256) # 💡全結合層用のバッチ正規化
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.prepare(x)        # 28x28
        x = self.inception1(x)     # 28x28
        x = self.pool(x)           # 14x14
        
        x = self.inception2(x)     # 14x14
        x = self.pool(x)           # 7x7
        
        x = x.view(-1, 128 * 7 * 7)
        x = F.relu(self.bn_fc1(self.fc1(x))) # 💡全結合層の直前でお掃除
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==========================================
# 3. 学習メイン処理
# ==========================================
def main():
    (x_train_np, t_train_np), (x_valid_np, t_valid_np) = load_train_data()

    x_train = torch.from_numpy(x_train_np).view(-1, 1, 28, 28).float()
    t_train = torch.from_numpy(t_train_np).long()
    x_valid = torch.from_numpy(x_valid_np).view(-1, 1, 28, 28).float()
    t_valid = torch.from_numpy(t_valid_np).long()

    train_dataset = TensorDataset(x_train, t_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    model = MiniGoogLeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # より細かく調整するため、8エポックごとに学習率を0.6倍にする設定に変更
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.6)

    epochs = 35 # 少しじっくり学習させるために35エポックに拡大
    print("限界突破：ガチ勢仕様Mini-GoogLeNetの学習をスタートします！")

    for epoch in range(epochs):
        model.train()  # 学習モード
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            
            # 💡 兵器②：データの壁を突破する「ランダム左右反転」（Data Augmentation）
            # 50%の確率で、このバッチの画像をすべて左右反転（次元3をひっくり返す）させます
            if torch.rand(1).item() > 0.5:
                images = torch.flip(images, dims=[3])

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = correct_train / total_train

        # 検証データでの精度評価
        model.eval()  # 評価モード（※重要：評価時は左右反転やバッチ正規化のサボりは無効になります）
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
    torch.save(model.state_dict(), "sample_weight.pkl")
    print("モデルを sample_weight.pkl に保存しました！")


if __name__ == "__main__":
    main()