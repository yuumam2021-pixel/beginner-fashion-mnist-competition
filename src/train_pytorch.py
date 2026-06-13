import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from load_fashion_mnist import load_train_data

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

class SuperResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.res1_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.res1_bn1 = nn.BatchNorm2d(128)
        self.res1_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.res1_bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.res2_conv1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.res2_bn1 = nn.BatchNorm2d(256)
        self.res2_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.res2_bn2 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(2, 2)
        
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

def main():
    print("データを読み込んでいます...")
    (x_train_np, t_train_np), (x_valid_np, t_valid_np) = load_train_data()

    x_valid = torch.from_numpy(x_valid_np).view(-1, 1, 28, 28).float()
    t_valid = torch.from_numpy(t_valid_np).long()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_valid = x_valid.to(device)
    t_valid_gpu = t_valid.to(device)

    # 💡 アンサンブル用に3つのモデルを順番に学習させます
    NUM_MODELS = 3

    for model_idx in range(NUM_MODELS):
        print(f"\n=======================================================")
        print(f"🚀 モデル {model_idx + 1}/{NUM_MODELS} の学習をスタート！")
        print(f"=======================================================")

        # モデルごとに「乱数の種」を変えて、それぞれ違う個性を持たせる
        torch.manual_seed(42 + model_idx)

        train_transform = T.Compose([
            T.RandomCrop(28, padding=4),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
        ])

        train_dataset = FashionMNISTDataset(x_train_np, t_train_np, transform=train_transform)
        # 💡 作戦2: バッチサイズを 64 に変更（より細かく学習）
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

        model = SuperResNet().to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        optimizer = optim.AdamW(model.parameters(), lr=0.0025, weight_decay=0.01)
        
        # 💡 作戦1: エポック数を 100 に変更（じっくり仕上げる）
        epochs = 100
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_valid_acc = 0.0
        # 保存名を変える（sample_weight_0.pkl, sample_weight_1.pkl...）
        best_weight_path = f"sample_weight_{model_idx}.pkl"

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            scheduler.step()
            epoch_loss = running_loss / len(train_loader.dataset)

            # 検証モード（TTA込み）
            model.eval()
            correct_valid = 0
            with torch.no_grad():
                outputs_orig = model(x_valid)
                outputs_flipped = model(torch.flip(x_valid, dims=[3]))
                outputs_final = (outputs_orig + outputs_flipped) / 2.0
                _, predicted_valid = torch.max(outputs_final, 1)
                correct_valid += (predicted_valid == t_valid_gpu).sum().item()
            
            valid_acc = correct_valid / t_valid_gpu.size(0)
            
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(model.state_dict(), best_weight_path)

            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {epoch_loss:.4f} | Valid Acc: {valid_acc:.4f} (Best: {best_valid_acc:.4f})")

        print(f"✨ モデル {model_idx + 1} の学習完了！ 最高Valid精度: {best_valid_acc:.4f} -> '{best_weight_path}' に保存しました。")

if __name__ == "__main__":
    main()