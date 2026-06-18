import time, os
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast


class AnimeDataset(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.data = []
        for author_dir in os.listdir(root):
            author_path = os.path.join(root, author_dir)
            for category in ['good', 'keep', 'demo1']:
                category_path = os.path.join(str(author_path), category)
                if os.path.exists(category_path):
                    for img_file in os.listdir(category_path):
                        img_path = os.path.join(category_path, img_file)
                        label = {'good': 2, 'keep': 1, 'demo1': 0}[category]
                        self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class KeepRatioResizePad:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        old_w, old_h = img.size
        ratio = min(self.target_size / old_w, self.target_size / old_h)
        new_w = int(old_w * ratio)
        new_h = int(old_h * ratio)

        img = transforms.functional.resize(img, (new_h, new_w))
        delta_w = self.target_size - new_w
        delta_h = self.target_size - new_h
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
        return transforms.functional.pad(img, padding, fill=255)


def get_train_transforms(input_size):
    train_transforms = transforms.Compose([
        KeepRatioResizePad(input_size),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_transforms


def get_val_transforms(input_size):
    val_transforms = transforms.Compose([
        KeepRatioResizePad(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return val_transforms


def create_model(device, class_nums):
    model = torchvision.models.efficientnet_b4(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, class_nums)
    model = model.to(device)
    return model


class Trainer:
    def __init__(self, input_size, data_root, batch_size=32, num_workers=4, patience=5, num_epochs=50, amp=True):
        self.input_size = input_size
        self.data_root = data_root
        self.patience = patience
        self.num_epochs = num_epochs
        self.amp = amp
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_dataset, self.val_dataset = self.prepare_datasets(input_size, data_root)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=num_workers,
                                       pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     pin_memory=True)
        self.train_model = create_model(self.device, 3)

    def prepare_datasets(self, input_size, data_root):

        full_dataset = AnimeDataset(data_root, get_train_transforms(input_size))

        # 划分训练验证集
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # 验证集使用独立的transform
        val_dataset.dataset.transform = get_val_transforms(input_size)

        return train_dataset, val_dataset

    def train(self):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.train_model.parameters()), lr=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

        best_acc = 0.0
        best_loss = float('inf')
        patience_counter = 0
        scaler = GradScaler()

        train_history = {'loss': [], 'acc': []}
        val_history = {'loss': [], 'acc': []}
        begin = time.time()
        for epoch in range(self.num_epochs):
            # 训练阶段
            self.train_model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Train]"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                if self.amp:
                    with autocast():
                        outputs = self.train_model(inputs)
                        loss = criterion(outputs, labels)

                    # 使用scaler缩放损失并反向传播
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.train_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                running_loss += loss.item() * inputs.size(0)

            train_loss = running_loss / len(self.train_loader.dataset)
            train_acc = correct / total
            train_history['loss'].append(train_loss)
            train_history['acc'].append(train_acc)

            # 验证阶段
            self.train_model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [Val]"):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.train_model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            val_loss = val_loss / len(self.val_loader.dataset)
            val_acc = correct / total
            val_history['loss'].append(val_loss)
            val_history['acc'].append(val_acc)

            # 学习率调整
            scheduler.step(val_loss)
            torch.save(self.train_model.state_dict(), f'models/demo0/demo0_v{epoch}.pth')
            # # 早停判断
            # if val_loss < best_loss:
            #     best_loss = val_loss
            #     best_acc = val_acc
            #     patience_counter = 0
            #     torch.save(self.train_model.state_dict(), '../best_model.pth')
            #     tqdm.write("saved best model")
            # else:
            #     patience_counter += 1

            print(f'time {round((time.time() - begin) / 60, 2)}min | Epoch {epoch + 1}/{self.num_epochs} | '
                  f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

            # if patience_counter >= self.patience:
            #     tqdm.write(f'Early stopping at epoch {epoch + 1}')
            #     break

        # 绘制训练曲线
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_history['loss'], label='Train Loss')
        plt.plot(val_history['loss'], label='Val Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_history['acc'], label='Train Acc')
        plt.plot(val_history['acc'], label='Val Acc')
        plt.legend()
        plt.savefig('training_curve.png')


def load_class_names(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    return [line.split(':')[1].strip() for line in sorted(lines, key=lambda x: int(x.split(':')[0]))]


class Predictor:
    def __init__(self, input_size, model_path, label_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = get_val_transforms(input_size)  # 使用验证时的预处理
        self.class_names = {'good': 2, 'keep': 1, 'trash': 0}

        # 加载模型
        self.model = create_model(self.device, 3)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def get_img_tensor(self, image_path):
        img = Image.open(image_path).convert('RGB')
        return self.transform(img).unsqueeze(0).to(self.device)

    def predict(self, img_tensor):
        with torch.no_grad():
            output = self.model(img_tensor)
            prob = torch.nn.functional.softmax(output, dim=1)

        pred_prob, pred_idx = torch.max(prob, dim=1)
        return {
            'class': self.class_names[pred_idx.item()],
            'probability': pred_prob.item(),
            'class_index': pred_idx.item()
        }

if __name__ == '__main__':
    trainer = Trainer(
        input_size=384,
        data_root='/media/bloodycrown/我的硬盘/Pictures/Storage/classification',
        batch_size=32,
        num_workers=8,
        patience=5,
        num_epochs=10,
        amp=True
    )
    trainer.train()