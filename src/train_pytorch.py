import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from load_fashion_mnist import load_train_data

# ==========================================
# 1. 究極進化：VGGスタイル4層CNNモデル
# ==========================================
class VGGStyle4LayerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 💡 【ブロック1】 画像サイズ 28x28 のまま、2回連続で畳み込み！
        # 1層目: 1 -> 32チャンネル
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 2層目: 32 -> 32チャンネル（さらに深く特徴を抽出）
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # ここで初めてサイズを半分にする（28x28 -> 14x14）
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        # 💡 【ブロック2】 画像サイズ 14x14 のまま、さらに2回連続で畳み込み！
        # 3層目: 32 -> 64チャンネル
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # 4層目: 64 -> 64チャンネル
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        # ここでサイズを半分にする（14x14 -> 7x7）
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        
        # 💡 全結合層（64チャンネル × 7マス × 7マス = 3136）
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(p=0.4) # 過学習防止のお守り

    def forward(self, x):
        # ブロック1の処理
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # ブロック2の処理
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # 1列に平坦化して全結合層へ
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==========================================
# 2. 学習メイン処理
# ==========================================
def main():
    (x_train_np, t_train_np), (x_valid_np, t_valid_np) = load_train_data()

    x_train = torch.from_numpy(x_train_np).view(-1, 1, 28, 28).float()
    t_train = torch.from_numpy(t_train_np).long()
    x_valid = torch.from_numpy(x_valid_np).view(-1, 1, 28, 28).float()
    t_valid = torch.from_numpy(t_valid_np).long()

    train_dataset = TensorDataset(x_train, t_train)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 💡 モデルを VGGStyle4LayerCNN に設定
    model = VGGStyle4LayerCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 8エポックごとに学習率を0.6倍にして、終盤の精度を微調整
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.6)

    epochs = 35
    print("限界突破：VGGスタイル4層CNNの学習をスタートします！")

    for epoch in range(epochs):
        model.train()  # 学習モード
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            
            # 💡 データ拡張：50%の確率で左右反転
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
    # 💡 システム指定通り sample_weight.pkl に保存
    torch.save(model.state_dict(), "sample_weight.pkl")
    print("モデルを sample_weight.pkl に保存しました！")


if __name__ == "__main__":
    main()