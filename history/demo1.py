import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import os
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast


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


class AnimeDataset(Dataset):
    def __init__(self, root_dir, author_to_idx, transform=None):
        self.root_dir = root_dir
        self.author_to_idx = author_to_idx
        self.transform = transform
        self.data = []
        self.labels = {'trash': 0, 'keep': 1, 'good': 2}

        for author_id in os.listdir(root_dir):
            author_path = os.path.join(root_dir, author_id)
            for label in os.listdir(author_path):
                label_path = os.path.join(author_path, label)
                for img_file in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_file)
                    self.data.append((img_path, author_id, self.labels[label]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, author_id, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        author_idx = self.author_to_idx.get(author_id, 0)
        return image, author_idx, label


class AnimeClassifier(nn.Module):
    def __init__(self, num_authors, embed_dim=64):
        super(AnimeClassifier, self).__init__()
        self.image_branch = models.efficientnet_b4(pretrained=True)
        self.image_branch.classifier = nn.Identity()
        self.author_embed = nn.Embedding(num_authors + 1, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(1792 + embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, image, author_idx):
        image_feat = self.image_branch(image)
        author_feat = self.author_embed(author_idx)
        combined_feat = torch.cat((image_feat, author_feat), dim=1)
        output = self.fc(combined_feat)
        return output


# 用于预测的自定义数据集类
class PredictionDataset(Dataset):
    def __init__(self, image_dir, author_id, author_to_idx, transform=None):
        self.image_dir = image_dir
        self.author_id = author_id
        self.author_to_idx = author_to_idx
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        author_idx = self.author_to_idx.get(self.author_id, 0)
        return image, author_idx, img_path


def train_model(model, train_loader, author_to_idx, criterion, optimizer, device, max_epochs):
    model.train()
    scaler = GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    for epoch in range(max_epochs):
        running_loss = 0.0
        for images, authors, labels in tqdm(train_loader):
            images, authors, labels = images.to(device), authors.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images, authors)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        # scheduler.step(val_loss)
        print(f'Epoch [{epoch + 1}/{max_epochs}], Train Loss: {train_loss:.4f}')
        torch.save({"state_dict": model.state_dict(), "author_to_idx": author_to_idx},f'models/demo1/demo1_v{epoch}.pth')

    return model


def load_model(model_path, num_authors, embed_dim=64, device='cuda'):
    """加载训练好的模型"""
    model = AnimeClassifier(num_authors=num_authors, embed_dim=embed_dim)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model


def predict_single(model, image_path, author_id, author_to_idx, transform, device):
    """单张图片预测"""
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    author_idx = torch.tensor([author_to_idx.get(author_id, 0)]).to(device)

    with torch.no_grad():
        output = model(image, author_idx)
        probabilities = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, 1)

    label_map = {0: 'trash', 1: 'keep', 2: 'good'}
    return {
        'prediction': label_map[pred.item()],
        'confidence': confidence.item(),
        'image_path': image_path
    }


def predict_batch(model, image_dir, author_id, author_to_idx, transform, device, batch_size=32, num_workers=4):
    """使用 Dataset 和 DataLoader 进行批量预测"""
    model.eval()
    results = []

    # 创建预测数据集
    pred_dataset = PredictionDataset(image_dir, author_id, author_to_idx, transform)
    pred_loader = DataLoader(
        pred_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    label_map = {0: 'trash', 1: 'keep', 2: 'good'}

    with torch.no_grad():
        for images, author_indices, img_paths in tqdm(pred_loader):
            images = images.to(device)
            author_indices = author_indices.to(device)

            outputs = model(images, author_indices)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probabilities, 1)

            for img_path, pred, conf in zip(img_paths, preds, confidences):
                results.append({
                    'prediction': label_map[pred.item()],
                    'confidence': conf.item(),
                    'image_path': img_path
                })

    return results


def get_author_mapping(root):
    author_to_idx = {}
    for idx, author_id in enumerate(os.listdir(root)):
        author_to_idx[author_id] = idx + 1
    return author_to_idx


if __name__ == '__main__':
    def main():
        root = '/media/bloodycrown/我的硬盘/Pictures/Storage/classification'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        author_to_idx = get_author_mapping(root)
        num_authors = len(author_to_idx)

        # 图片预处理（训练用）
        train_transform = transforms.Compose([
            KeepRatioResizePad(384),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


        # 训练部分
        dataset = AnimeDataset(root_dir=root, author_to_idx=author_to_idx, transform=train_transform)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

        model = AnimeClassifier(num_authors=num_authors, embed_dim=64).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


        # 取消注释以下行以训练模型
        train_model(model, train_loader, author_to_idx, criterion, optimizer, device, max_epochs=10)

    main()
