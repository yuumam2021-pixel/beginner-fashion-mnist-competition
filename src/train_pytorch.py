import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from load_fashion_mnist import load_train_data

# ==========================================
# 1. 超高速・高精度モデル：Fast ResNet (ResNet-9スタイル)
# ==========================================
class FastResNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 【準備層】 28x28 のまま特徴を抽出
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 【層1】 14x14 に縮小
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 💡 【スキップ構造 1】 14x14のまま、近道を使って深く学習
        self.res1_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.res1_bn1 = nn.BatchNorm2d(64)
        self.res1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.res1_bn2 = nn.BatchNorm2d(64)
        
        # 【層2】 7x7 に縮小
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 💡 【スキップ構造 2】 7x7のまま、近道を使ってさらに深く学習
        self.res2_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res2_bn1 = nn.BatchNorm2d(128)
        self.res2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.res2_bn2 = nn.BatchNorm2d(128)
        
        # 【仕上げ】 空間情報（7x7）を維持したまま全結合層へ
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        # 準備
        x = F.relu(self.bn1(self.conv1(x)))
        
        # 層1
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        
        # スキップ構造1（足し算！）
        r = F.relu(self.res1_bn1(self.res1_conv1(x)))
        x = x + self.res1_bn2(self.res1_conv2(r))
        x = F.relu(x)
        
        # 層2
        x = self.pool2(F.relu(self.bn3(self.conv3(x))))
        
        # スキップ構造2（足し算！）
        r = F.relu(self.res2_bn1(self.res2_conv1(x)))
        x = x + self.res2_bn2(self.res2_conv2(r))
        x = F.relu(x)
        
        # 分類
        return self.classifier(x)

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

    # 💡 モデルを FastResNet に変更
    model = FastResNet()
    criterion = nn.CrossEntropyLoss()
    
    # 💡 最も安定して速い通常の Adam に戻す
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # スケジューラー（10エポックごとに0.5倍）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epochs = 30 # 爆速なのでサクサク進みます
    print("限界突破：超高速 Fast ResNet の学習をスタートします！")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            # データ拡張（左右反転）は精度向上に効くので残します
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

        model.eval()
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