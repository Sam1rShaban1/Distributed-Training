import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """Sets up the distributed environment."""
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']

    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{master_addr}:{master_port}',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)
    return rank, world_size

def cleanup():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()

def main():
    rank, world_size = setup_distributed()
    print(f"Starting training on rank {rank}/{world_size}.")

    # 1. Model: Simple Linear Regression
    model = nn.Linear(10, 1).to(rank)
    # Wrap model with DDP
    ddp_model = DDP(model, device_ids=[rank])

    # 2. Data: Create synthetic data
    # Each rank will generate the same data, but the sampler will divide it.
    inputs = torch.randn(1000, 10)
    labels = inputs.sum(dim=1, keepdim=True) + torch.randn(1000, 1) # y = sum(x) + noise
    dataset = TensorDataset(inputs, labels)

    # 3. Sampler and DataLoader
    # DistributedSampler ensures each process gets a different slice of the data
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # 4. Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

    # 5. Training Loop
    for epoch in range(10):
        # The sampler needs the epoch to shuffle data properly
        sampler.set_epoch(epoch)
        for batch_inputs, batch_labels in dataloader:
            batch_inputs = batch_inputs.to(rank)
            batch_labels = batch_labels.to(rank)

            optimizer.zero_grad()
            outputs = ddp_model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        # Only print loss from the master process
        if rank == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    print(f"Rank {rank} finished training.")
    cleanup()

if __name__ == '__main__':
    main()