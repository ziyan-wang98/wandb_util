from example_config import get_config
from wandb_utils import init_wandb, finish_wandb
import wandb
import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':
    # Parse command-line arguments
    args = get_config()

    # Initialize WandB
    init_wandb(args)

    # Set up model, optimizer, loss function
    model = nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(10):
        # Generate random data
        x = torch.randn(100, 1)
        y = 2 * x + 1

        # Forward pass
        y_pred = model(x)

        # Compute loss
        loss = loss_fn(y_pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log metrics to WandB
        wandb.log({'epoch': epoch, 'loss': loss.item()})
    
    finish_wandb(args)
