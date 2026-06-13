import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from load_fashion_mnist import load_train_data

# ==========================================
# 0. カスタムデータセット（強力なデータ拡張）
# ==========================================
class FashionMNISTDataset(Dataset):
    def __init__(self, x_np, t_np, transform=None):
        self.x = torch.from_numpy(x_np).view(-1, 1, 28, 28).float()
        self.t = torch.from_numpy(t_np).long()
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.t[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ==========================================
# 1. 超高精度・大規模モデル：SuperResNet (512ch)
# ==========================================
class SuperResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 準備層 (1 -> 64チャンネルへ拡大)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 第1層 (128chに拡大 + 縮小)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2) # 14x14
        
        # 残差ブロック1
        self.res1_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.res1_bn1 = nn.BatchNorm2d(128)
        self.res1_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.res1_bn2 = nn.BatchNorm2d(128)
        
        # 第2層 (256chに拡大 + 縮小)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2) # 7x7
        
        # 残差ブロック2
        self.res2_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.res2_bn1 = nn.BatchNorm2d(256)
        self.res2_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.res2_bn2 = nn.BatchNorm2d(256)

        # 第3層 (512chに拡大 + 最終縮小)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2, 2) # 3x3
        
        # 仕上げ (GAP + 強力なDropoutで過学習を徹底ガード)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        
        r = F.relu(self.res1_bn1(self.res1_conv1(x)))
        x = x + self.res1_bn2(self.res1_conv2(r))
        x = F.relu(x)
        
        x = self.pool2(F.relu(self.bn3(self.conv3(x))))
        
        r = F.relu(self.res2_bn1(self.res2_conv1(x)))
        x = x + self.res2_bn2(self.res2_conv2(r))
        x = F.relu(x)

        x = self.pool3(F.relu(self.bn4(self.conv4(x))))
        
        x = self.gap(x)
        return self.classifier(x)

# ==========================================
# 2. 学習メイン処理
# ==========================================
def main():
    print("データを読み込んでいます...")
    (x_train_np, t_train_np), (x_valid_np, t_valid_np) = load_train_data()

    # 🔥 97%突破のための最強データ拡張（ズレ・反転・さらに「回転」を追加）
    train_transform = T.Compose([
        T.RandomCrop(28, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10), # 10度以内のランダムな傾きに対応
    ])

    train_dataset = FashionMNISTDataset(x_train_np, t_train_np, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    x_valid = torch.from_numpy(x_valid_np).view(-1, 1, 28, 28).float()
    t_valid = torch.from_numpy(t_valid_np).long()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 使用するデバイス: {device}")

    model = SuperResNet().to(device)
    
    # 💡 97%最適化：Label Smoothingを 0.05 に微調整
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # AdamWの学習率を 0.0025 に最適化
    optimizer = optim.AdamW(model.parameters(), lr=0.0025, weight_decay=0.01)
    
    # じっくり50エポック回して極限まで収束させます
    epochs = 50
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("👑 ターゲット：Valid Acc 97%突破ミッション スタート！")

    best_valid_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

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

        # -----------------------------
        # 検証モード（TTA込み）
        # -----------------------------
        model.eval()
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            images_valid = x_valid.to(device)
            labels_valid_gpu = t_valid.to(device)
            
            # TTA: オリジナルと左右反転の予測をアンサンブル
            outputs_orig = model(images_valid)
            images_flipped = torch.flip(images_valid, dims=[3])
            outputs_flipped = model(images_flipped)
            
            outputs_final = (outputs_orig + outputs_flipped) / 2.0
            
            _, predicted_valid = torch.max(outputs_final, 1)
            total_valid += labels_valid_gpu.size(0)
            correct_valid += (predicted_valid == labels_valid_gpu).sum().item()
        
        valid_acc = correct_valid / total_valid
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            # 最高精度を更新したらモデルを保存
            torch.save(model.state_dict(), "sample_weight.pkl")

        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f} | Valid Acc: {valid_acc:.4f} (Best: {best_valid_acc:.4f})")

    print(f"\n✨ 学習完了！ 最高Valid精度: {best_valid_acc:.4f}")

if __name__ == "__main__":
    main()