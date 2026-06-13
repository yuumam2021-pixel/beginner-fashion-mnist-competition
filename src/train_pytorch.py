import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from load_fashion_mnist import load_train_data

# ==========================================
# 0. カスタムデータセット（強力なデータ拡張を使うため）
# ==========================================
class FashionMNISTDataset(Dataset):
    def __init__(self, x_np, t_np, transform=None):
        # Numpy配列をPyTorchのTensorに変換
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
# 1. 超高速・高精度モデル：Fast ResNet
# ==========================================
class FastResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 準備層
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # 第1層 (縮小)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # スキップ構造1
        self.res1_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.res1_bn1 = nn.BatchNorm2d(64)
        self.res1_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.res1_bn2 = nn.BatchNorm2d(64)
        
        # 第2層 (縮小)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # スキップ構造2
        self.res2_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.res2_bn1 = nn.BatchNorm2d(128)
        self.res2_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.res2_bn2 = nn.BatchNorm2d(128)
        
        # 仕上げ (GAPを使ってパラメータ削減＆過学習防止)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
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
        
        x = self.gap(x)
        return self.classifier(x)

# ==========================================
# 2. 学習メイン処理 (全部盛り版)
# ==========================================
def main():
    print("データを読み込んでいます...")
    (x_train_np, t_train_np), (x_valid_np, t_valid_np) = load_train_data()

    # 💡 秘伝のタレ2：訓練用の強力なデータ拡張（少しパディングしてランダムに切り取る ＋ 左右反転）
    train_transform = T.Compose([
        T.RandomCrop(28, padding=4),
        T.RandomHorizontalFlip(p=0.5),
    ])

    train_dataset = FashionMNISTDataset(x_train_np, t_train_np, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    # 検証データはTensorDatasetのまま（データ拡張しないため）
    x_valid = torch.from_numpy(x_valid_np).view(-1, 1, 28, 28).float()
    t_valid = torch.from_numpy(t_valid_np).long()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 使用するデバイス: {device}")

    model = FastResNet().to(device)
    
    # 💡 秘伝のタレ5：ラベルスムージング（正解を100%ではなく90%として学習させ、過学習を防ぐ）
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 💡 秘伝のタレ3：AdamW（通常のAdamより過学習に強い）
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    
    epochs = 40 # 少し長めに回します
    
    # 💡 秘伝のタレ4：Cosine Annealing（学習率をコサインカーブのように滑らかに下げる）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("限界突破：最高精度を狙うフルカスタム学習をスタートします！")

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
        # 検証モード（TTAの実装）
        # -----------------------------
        model.eval()
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            images_valid = x_valid.to(device)
            labels_valid_gpu = t_valid.to(device)
            
            # TTA 1：オリジナルの画像
            outputs_orig = model(images_valid)
            
            # TTA 2：左右反転画像
            images_flipped = torch.flip(images_valid, dims=[3])
            outputs_flipped = model(images_flipped)
            
            # TTA 3：アンサンブル（平均）
            outputs_final = (outputs_orig + outputs_flipped) / 2.0
            
            _, predicted_valid = torch.max(outputs_final, 1)
            total_valid += labels_valid_gpu.size(0)
            correct_valid += (predicted_valid == labels_valid_gpu).sum().item()
        
        valid_acc = correct_valid / total_valid

        print(f"Epoch {epoch+1:02d} | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.4f} | Valid Acc: {valid_acc:.4f}")

    print("学習が完了しました！")
    torch.save(model.state_dict(), "sample_weight.pkl")
    print("モデルを sample_weight.pkl に保存しました！")

if __name__ == "__main__":
    main()