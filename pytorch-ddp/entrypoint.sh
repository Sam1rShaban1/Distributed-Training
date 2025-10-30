#!/bin/bash

# Wait for the master node to be ready
if [ "$RANK" -eq 0 ]; then
  echo "Waiting for other workers to join..."
  sleep 5
fi

# Launch the training script using torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=2 --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train_ddp.py
