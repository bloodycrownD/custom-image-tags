import os
from collections import Counter  # For counting labels

import torch, threading
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split  # Import Subset
from tqdm import tqdm


class KeepRatioResizePad:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        old_w, old_h = img.size
        # Handle potential zero dimensions if image loading fails or is corrupted
        if old_w == 0 or old_h == 0:
            # Return a dummy black image or raise an error
            print(f"Warning: Image with zero dimension encountered.")
            return Image.new('RGB', (self.target_size, self.target_size), color='black')

        ratio = min(self.target_size / old_w, self.target_size / old_h)
        new_w = max(1, int(old_w * ratio))  # Ensure dimensions are at least 1
        new_h = max(1, int(old_h * ratio))  # Ensure dimensions are at least 1
        img = transforms.functional.resize(img, (new_h, new_w))
        delta_w = self.target_size - new_w
        delta_h = self.target_size - new_h
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
        return transforms.functional.pad(img, padding, fill=255)  # fill=0 is black padding


from concurrent.futures import ThreadPoolExecutor
import threading

class CacheDataset(Dataset):
    def __init__(self, root_dir, author_to_idx, transform=None):
        self.root_dir = root_dir
        self.author_to_idx = author_to_idx
        self.transform = transform
        self.data = []
        self.cache = {}
        self.load = set()  # 使用 set 提高查找效率
        self.labels_map = {'trash': 0, 'keep': 1, 'good': 2}
        self.lock = threading.Lock()
        self.cache_idx = 2000
        self.executor = ThreadPoolExecutor(max_workers=4)  # 限制最大线程数
        print("Loading dataset...")
        for author_id in tqdm(os.listdir(root_dir)):
            author_path = os.path.join(root_dir, author_id)
            if not os.path.isdir(author_path):
                continue
            for label in os.listdir(author_path):
                label_path = os.path.join(author_path, label)
                if not os.path.isdir(label_path) or label not in self.labels_map:
                    continue
                for img_file in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_file)
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        self.data.append((img_path, author_id, self.labels_map[label]))
                    else:
                        print(f"Skipping non-image file: {img_path}")
        print(f"Dataset loaded with {len(self.data)} samples.")
        self.init_cache()

    def init_cache(self):
        print("init cache")
        for idx in tqdm(range(min(len(self.data), self.cache_idx))):
            img_path, author_id, label = self.data[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            author_idx = self.author_to_idx.get(author_id, 0)
            self.cache[idx] = (image, author_idx, label)

    def get_item_from_disk(self, idx):
        with self.lock:
            img_path, author_id, label = self.data[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            author_idx = self.author_to_idx.get(author_id, 0)
            return image, author_idx, label

    def preload_to_cache(self, idx):
        with self.lock:
            if idx not in self.cache and idx not in self.load:
                self.load.add(idx)
                image, author_idx, label = self.get_item_from_disk(idx)
                self.cache[idx] = (image, author_idx, label)
                self.load.remove(idx)

    def __getitem__(self, idx):
        with self.lock:
            self.load.add(idx)
        if idx in self.cache:
            with self.lock:
                image, author_idx, label = self.cache[idx]
                del self.cache[idx]
                self.load.remove(idx)
            # 异步预加载下一个索引
            next_idx = (idx + 1) % len(self.data)  # 循环索引
            self.executor.submit(self.preload_to_cache, next_idx)
            return image, author_idx, label
        else:
            image, author_idx, label = self.get_item_from_disk(idx)
            with self.lock:
                self.load.remove(idx)
            return image, author_idx, label

    def __len__(self):
        return len(self.data)

    def get_labels(self):
        return [label for _, _, label in self.data]

    def __del__(self):
        self.executor.shutdown(wait=True)  # 清理线程池

class AnimeDataset(Dataset):
    def __init__(self, root_dir, author_to_idx, transform=None):
        self.root_dir = root_dir
        self.author_to_idx = author_to_idx
        self.transform = transform
        self.data = []
        self.labels_map = {'trash': 0, 'keep': 1, 'good': 2}

        print("Loading dataset...")
        for author_id in tqdm(os.listdir(root_dir)):
            author_path = os.path.join(root_dir, author_id)
            if not os.path.isdir(author_path): continue  # Skip files, only process directories
            for label in os.listdir(author_path):
                label_path = os.path.join(author_path, label)
                if not os.path.isdir(
                        label_path) or label not in self.labels_map: continue  # Skip files or invalid label folders
                for img_file in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_file)
                    # Basic check for image file extensions
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        self.data.append((img_path, author_id, self.labels_map[label]))
                    else:
                        print(f"Skipping non-image file: {img_path}")
        print(f"Dataset loaded with {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, author_id, label = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Returning black image.")
            # Return a dummy tensor or handle appropriately
            image = Image.new('RGB', (512, 512), color='black')  # Assuming target size 512 for dummy
            image = transforms.ToTensor()(image)  # Basic transform if none provided
            author_idx = self.author_to_idx.get(author_id, 0)  # Use 0 for unknown/error case
            return image, author_idx, label

        # Apply transformation if specified
        image = self.transform(image)
        author_idx = self.author_to_idx.get(author_id, 0)
        return image, author_idx, label

    def get_labels(self):
        # Helper function to get all labels for calculating weights
        return [label for _, _, label in self.data]


class AnimeClassifier(nn.Module):
    def __init__(self, num_authors, embed_dim=64, num_classes=3):  # Added num_classes
        super(AnimeClassifier, self).__init__()
        # Consider using a smaller EfficientNet (e.g., b0, b2) if B4 is too slow or overfits
        self.image_branch = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.DEFAULT)  # Use updated weights API
        num_img_features = self.image_branch.classifier[1].in_features  # Get features before final layer
        self.image_branch.classifier = nn.Identity()  # Remove original classifier
        # Embedding for author ID. +1 for potential unknown authors (index 0)
        self.author_embed = nn.Embedding(num_authors + 1, embed_dim)
        # Classifier head
        self.fc = nn.Sequential(
            nn.BatchNorm1d(num_img_features + embed_dim),  # Add BatchNorm
            nn.Linear(num_img_features + embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Add Dropout
            nn.Linear(512, num_classes)  # Use num_classes
        )

    def forward(self, image, author_idx):
        image_feat = self.image_branch(image)
        author_feat = self.author_embed(author_idx)
        combined_feat = torch.cat((image_feat, author_feat), dim=1)
        output = self.fc(combined_feat)
        return output


class PredictionDataset(Dataset):
    def __init__(self, image_dir, author_id, author_to_idx, transform=None):
        self.image_dir = image_dir
        self.author_id = author_id
        self.author_to_idx = author_to_idx
        self.transform = transform
        self.image_paths = []
        print(f"Loading prediction images from: {image_dir}")
        if os.path.isdir(image_dir):
            self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if
                                f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        print(f"Found {len(self.image_paths)} images for prediction.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading prediction image {img_path}: {e}. Returning black image.")
            image = Image.new('RGB', (512, 512), color='black')  # Dummy image
            # Apply minimal transform
            temp_transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image = temp_transform(image)

        image = self.transform(image)

        # Use 0 if author_id is not found in the training map, though it should be provided.
        author_idx = self.author_to_idx.get(self.author_id, 0)
        return image, author_idx, img_path


# --- Evaluation Function ---
def evaluate_model(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculation
        for images, authors, labels in tqdm(dataloader, desc="Validation"):
            images, authors, labels = images.to(device), authors.to(device), labels.to(device)

            # No autocast needed for evaluation usually, unless required by model layers
            outputs = model(images, authors)
            loss = criterion(outputs, labels)  # Use the same criterion (even if weighted for training)

            running_loss += loss.item() * images.size(0)  # Accumulate loss weighted by batch size
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    val_loss = running_loss / total_samples
    val_accuracy = correct_predictions / total_samples
    return val_loss, val_accuracy


# --- Updated Training Function ---
def train_model(model, train_loader, val_loader, author_to_idx, criterion, optimizer, scheduler, device, max_epochs,
                early_stopping_patience=5):
    model.train()
    scaler = GradScaler()  # For mixed precision
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        best_model_path = f"models/demo2/demo2_v{epoch}.pth"
        for images, authors, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs} Training"):
            images, authors, labels = images.to(device), authors.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():  # Mixed precision
                outputs = model(images, authors)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)  # Use dataset length for average loss per sample
        train_accuracy = train_correct / train_total

        # Validation step
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        print(f'Epoch [{epoch + 1}/{max_epochs}], Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        torch.save({
            "state_dict": model.state_dict(),
            "author_to_idx": author_to_idx,
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        }, best_model_path)
        # Learning rate scheduler step (based on validation loss)
        scheduler.step(val_loss)
        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            print(f"Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}). Saving model...")
            best_val_loss = val_loss
            # Save model state and author mapping

            epochs_no_improve = 0  # Reset counter
        else:
            epochs_no_improve += 1
            print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement.")
            break


# --- Load Model Function (Updated to handle new save format) ---
def load_model_from_checkpoint(model_path, num_authors, embed_dim=64, num_classes=3, device='cuda'):
    """Loads model and author mapping from a checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    author_to_idx = checkpoint['author_to_idx']
    # Recreate model structure
    model = AnimeClassifier(num_authors=num_authors, embed_dim=embed_dim, num_classes=num_classes)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()  # Set to evaluation mode
    print(
        f"Model loaded from {model_path}. Trained for {checkpoint.get('epoch', 'N/A')} epochs. Best Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    return model, author_to_idx


# --- predict_single (Updated to use loaded author_to_idx) ---
def predict_single(model, image_path, author_id, author_to_idx, transform, device):
    """Single image prediction using the loaded model and author_to_idx."""
    model.eval()
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None  # Or return an error structure

    image = transform(image).unsqueeze(0).to(device)
    # Use 0 if author_id not in the map from training
    author_idx_val = author_to_idx.get(author_id, 0)
    if author_idx_val == 0:
        print(f"Warning: Author ID '{author_id}' not found in training data mapping. Using index 0.")
    author_idx = torch.tensor([author_idx_val]).to(device)

    label_map = {0: 'trash', 1: 'keep', 2: 'good'}  # Ensure consistency

    with torch.no_grad():
        output = model(image, author_idx)
        probabilities = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probabilities, 1)

    return {
        'prediction': label_map[pred.item()],
        'confidence': confidence.item(),
        'image_path': image_path
    }


# --- predict_batch (Updated to use loaded author_to_idx) ---
def predict_batch(model, image_dir, author_id, author_to_idx, transform, device, batch_size=32, num_workers=4):
    """Batch prediction using the loaded model and author_to_idx."""
    model.eval()
    results = []

    pred_dataset = PredictionDataset(image_dir, author_id, author_to_idx, transform)
    if len(pred_dataset) == 0:
        print("No valid images found for prediction in the specified directory.")
        return results

    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    label_map = {0: 'trash', 1: 'keep', 2: 'good'}  # Ensure consistency

    with torch.no_grad():
        for images, author_indices, img_paths in tqdm(pred_loader, desc="Batch Prediction"):
            images = images.to(device)
            author_indices = author_indices.to(device)  # Already contains correct indices from PredictionDataset

            outputs = model(images, author_indices)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, preds = torch.max(probabilities, 1)

            for img_path, pred, conf in zip(img_paths, preds, confidences):
                # Check if img_path is valid (might be None if PredictionDataset had errors)
                if img_path:
                    results.append({
                        'prediction': label_map[pred.item()],
                        'confidence': conf.item(),
                        'image_path': img_path
                    })

    return results


# --- get_author_mapping (No changes needed) ---
def get_author_mapping(root_dir):
    author_to_idx = {}
    for idx, author_id in enumerate(os.listdir(root_dir)):
        author_to_idx[author_id] = idx + 1
    return author_to_idx


if __name__ == '__main__':
    def main():
        # --- Configuration ---
        root_dir = '/media/bloodycrown/我的硬盘/Pictures/Storage/classification'
        target_image_size = 384  # Try a smaller size first (EfficientNet B4 default is 384)
        batch_size = 32  # Reduce batch size if memory issues occur with larger images/model
        learning_rate = 5e-5  # Potentially smaller LR for fine-tuning
        max_epochs = 10  # Increase epochs, rely on early stopping
        early_stopping_patience = 7  # Stop if no improvement for 7 epochs
        num_workers = 8
        train_val_split_ratio = 0.85  # 80% for training, 20% for validation
        embed_dim = 64  # Experiment with embedding dimension
        num_classes = 3  # trash, keep, good

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # --- Data Preparation ---
        author_to_idx = get_author_mapping(root_dir)
        if not author_to_idx:
            print("Could not generate author mapping. Exiting.")
            return
        num_authors = len(author_to_idx)

        # Transforms
        train_transform = transforms.Compose([
            KeepRatioResizePad(target_image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_pred_transform = transforms.Compose([  # Same transform for validation and prediction
            KeepRatioResizePad(target_image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Create full dataset
        full_dataset = AnimeDataset(root_dir=root_dir, author_to_idx=author_to_idx,
                                    transform=train_transform)  # Use train transform for full initially
        if len(full_dataset) == 0:
            print("Dataset is empty. Check the root directory and data structure.")
            return

        # Split dataset
        total_size = len(full_dataset)
        train_size = int(total_size * train_val_split_ratio)
        val_size = total_size - train_size
        print(f"Splitting dataset: {train_size} train, {val_size} validation samples.")
        # Ensure reproducibility if desired: generator=torch.Generator().manual_seed(42)
        train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

        # Important: Apply the correct transform to the validation subset
        # We need to wrap the subset to change its transform attribute.
        # A bit clumsy, but common practice.
        val_subset.dataset = AnimeDataset(root_dir=root_dir, author_to_idx=author_to_idx,
                                          transform=val_pred_transform)  # Recreate underlying with correct transform
        # The indices remain the same from random_split
        # train_subset already has the correct transform via full_dataset

        # --- Calculate Class Weights for Training Set ---
        print("Calculating class weights for training set...")
        train_labels = [full_dataset.data[i][2] for i in
                        train_subset.indices]  # Get labels only for the training indices
        label_counts = Counter(train_labels)
        print(f"Training label distribution: {label_counts}")

        # Handle case where a class might be missing in the training split (rare but possible)
        weights = [0.0] * num_classes
        total_train_samples = len(train_labels)
        for i in range(num_classes):
            if label_counts[i] > 0:
                weights[i] = total_train_samples / (num_classes * label_counts[i])  # Inverse frequency weighting
            # else: handle zero count if necessary (e.g., assign a very small weight or 1.0)

        class_weights = torch.tensor(weights, dtype=torch.float).to(device)
        print(f"Calculated class weights: {class_weights}")

        # DataLoaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                pin_memory=True)

        # --- Model, Loss, Optimizer ---
        model = AnimeClassifier(num_authors=num_authors, embed_dim=embed_dim, num_classes=num_classes).to(device)

        # Use weighted loss
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                      weight_decay=1e-2)  # Use AdamW with weight decay
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3,
                                      verbose=True)  # Reduce LR significantly on plateau
        train_model(
            model, train_loader, val_loader, author_to_idx,
            criterion, optimizer, scheduler, device, max_epochs, early_stopping_patience
        )


    main()
