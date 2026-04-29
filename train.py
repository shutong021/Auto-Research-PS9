import torch
import torch.nn as nn
import torch.optim as optim
from prepare import train_loader, test_loader, evaluate, log_experiment
import time
from torchsummary import summary

# ==================== Baseline Model (Simple CNN) ====================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.net(x)

# ==================== Training Function ====================
def train_model():
    model = SimpleCNN().to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    # Train
    model.train()
    for epoch in range(15):   # max 15 epochs
        for images, labels in train_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    train_time = time.time() - start_time
    val_acc = evaluate(model, test_loader)
    param_count = sum(p.numel() for p in model.parameters())
    
    print(f"Validation Accuracy: {val_acc:.4f}% | Time: {train_time:.2f}s | Params: {param_count}")
    
    return val_acc, train_time, param_count, model

# ==================== Run Experiment ====================
if __name__ == "__main__":
    print("Starting experiment...")
    val_acc, train_time, param_count, model = train_model()
    
    # For first run, we consider it the baseline
    log_experiment(
        idea="Baseline SimpleCNN (2 conv + 2 fc)",
        val_acc=val_acc,
        train_time=train_time,
        param_count=param_count,
        best_so_far=val_acc,
        kept=True,
        notes="Initial baseline"
    )
    
    print("Experiment completed.")
