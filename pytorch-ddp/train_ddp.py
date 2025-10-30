import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP

# Simple Neural Network Model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(rank, world_size):
    # Initialize DDP
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device for this rank
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Create model and move it to the appropriate device
    model = SimpleNN().to(device)
    model = DDP(model, device_ids=[rank])

    # Create a simple dataset
    data = torch.randn(100, 10)
    targets = torch.randint(0, 2, (100,))

    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train loop
    for epoch in range(5):  # 5 epochs
        model.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if rank == 0:  # Print loss from master node
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Cleanup
    dist.destroy_process_group()

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    train(rank, world_size)

if __name__ == "__main__":
    main()
