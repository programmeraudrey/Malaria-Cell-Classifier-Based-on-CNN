import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import sys  # 导入sys模块
from datetime import datetime

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(1000)

# Configuration
data_root_dir = '/root/autodl-tmp/data/'
image_directory = data_root_dir + 'cell_images/'
SIZE = 64
BATCH_SIZE = 64
EPOCHS = 100

# Dataset and DataLoader
class MalariaDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# Model definition
class MalariaCNN(nn.Module):
    def __init__(self, dropout = 0.2):
        super(MalariaCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32*(SIZE//4)*(SIZE//4), 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Prepare dataset
def prepare_datasets(random_sample_ratio = 1.0, transform = True):
    parasitized_dir = os.path.join(image_directory, 'Parasitized')
    uninfected_dir = os.path.join(image_directory, 'Uninfected')

    parasitized_paths = [os.path.join(parasitized_dir, f) for f in os.listdir(parasitized_dir) if f.endswith('.png')]
    uninfected_paths = [os.path.join(uninfected_dir, f) for f in os.listdir(uninfected_dir) if f.endswith('.png')]

    all_paths = parasitized_paths + uninfected_paths
    labels = [0]*len(parasitized_paths) + [1]*len(uninfected_paths)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_paths, labels, test_size=0.2, random_state=0
    )

    # Split the training data into 90% train and 10% validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.1, random_state=0
    )

    train_paths, _, train_labels, _ = train_test_split(
        train_paths, train_labels, test_size = random_sample_ratio, random_state=0
    )

    # 定义数据增强和归一化
    if transform:
        train_transform = transforms.Compose([
            transforms.Resize((SIZE, SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_test_transform = transforms.Compose([
            transforms.Resize((SIZE, SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = MalariaDataset(train_paths, train_labels, train_transform)
        val_dataset = MalariaDataset(val_paths, val_labels, val_test_transform)
        test_dataset = MalariaDataset(test_paths, test_labels, val_test_transform)
    else:
        transform = transforms.Compose([
            transforms.Resize((SIZE, SIZE)),
            transforms.ToTensor(),
        ])

        train_dataset = MalariaDataset(train_paths, train_labels, transform)
        val_dataset = MalariaDataset(val_paths, val_labels, transform)
        test_dataset = MalariaDataset(test_paths, test_labels, transform)

    return train_dataset, val_dataset, test_dataset

# Training function
def train_model(model, train_loader, val_loader, device, ckpt_path, lr = 0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = 100 * val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}]")
        print(f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        print(f"Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.2f}%\n")

        if val_epoch_acc > best_val_acc:
            torch.save(model.state_dict(), ckpt_path)
            best_val_acc = val_epoch_acc
            best_epoch = epoch
            print(f"New best model saved with Val Acc at epoch {epoch}: {best_val_acc:.2f}%\n")
    
    return train_losses, train_accs, val_losses, val_accs, best_epoch

# Test function
def test_model(model, test_loader, device, best_epoch):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    test_loss = test_loss / len(test_loader)
    print(f"Epoch {best_epoch} Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Main execution
if __name__ == "__main__":

    lr = 0.001
    dropout = 0.5
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
     
    arg = f"lr_{lr}_dropout_{dropout}_bz_{BATCH_SIZE}"

    # 设置可见的GPU（假设实际物理卡号为1）
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 这里填写物理GPU编号
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not os.path.exists("./logs/"):
        os.makedirs("./logs/")
    if not os.path.exists("./checkpoints/"):
        os.makedirs("./checkpoints/")
    if not os.path.exists("./pngs/"):
        os.makedirs("./pngs/")

    # 重定向标准输出到日志文件
    log_path = f'./logs/t0_training_log_transform_{arg}_{filename_time}.txt'
    sys.stdout = Logger(log_path)

    for random_sample_ratio in ratios:
        print(f"\n=== Training with {random_sample_ratio*100}% of training data ===")
        args = f"sample_{random_sample_ratio}_"+ arg
        ckpt_path = f'./checkpoints/t0_best_malaria_cnn_{args}_{filename_time}.pth'
        
        # Prepare data
        train_dataset, val_dataset, test_dataset = prepare_datasets(random_sample_ratio)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, drop_last=True)
        
        # Initialize model
        model = MalariaCNN(dropout = dropout).to(device)
        # print(model)  # 模型结构会被记录到日志

        # Train model
        train_loss, train_acc, val_loss, val_acc, best_epoch = train_model(model, train_loader, val_loader, device, ckpt_path, lr)

        model.load_state_dict(torch.load(ckpt_path))
        # Test model
        print(f"lr = {lr}")
        print(f"dropout = {dropout}")
        print(f"BATCH_SIZE={BATCH_SIZE}")
        
        test_model(model, test_loader, device, best_epoch)
        
        # Plot results
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_acc, label='Train Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"./pngs/t0_training_curv_{args}_{filename_time}.png")