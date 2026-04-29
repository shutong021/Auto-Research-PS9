import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import pandas as pd
from datetime import datetime

# ------------------- Data Loading (Read-only) -------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

# ------------------- Evaluation Function (Read-only) -------------------
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# ------------------- Logging -------------------
def log_experiment(idea, val_acc, train_time, param_count, best_so_far, kept, notes=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exp_id = len(pd.read_csv('results.tsv', sep='\t')) if pd.io.common.file_exists('results.tsv') else 0
    
    new_row = {
        'experiment_id': exp_id,
        'timestamp': timestamp,
        'idea': idea,
        'val_accuracy': round(val_acc, 4),
        'train_time_sec': round(train_time, 2),
        'param_count': param_count,
        'best_so_far': round(best_so_far, 4),
        'kept': kept,
        'notes': notes
    }
    
    df = pd.DataFrame([new_row])
    df.to_csv('results.tsv', sep='\t', mode='a', header=not pd.io.common.file_exists('results.tsv'), index=False)
    print(f"Logged: {idea} | Val Acc: {val_acc:.4f}% | Kept: {kept}")
