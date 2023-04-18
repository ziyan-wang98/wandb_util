# How to use Wandb (Wandb_util)
This repository contains a simple script for showing how to use WandB to log metrics during training and visualize the results in the WandB dashboard.

## Requirements
* python >= 3.6
* wandb

You can install the required Python packages using pip:

```bash
pip install torch wandb
```

## Quick test
1. Clone this repository and setup your WandB API key in `example_config.py`:
2. Run the example script:
    ```bash
    python main.py
    ```

## How to use in your project
1. Copy `wandb_util.py` to your project directory.
2. Add Wandb related parsers to your config file.
2. Add the following lines to your training script:
    ```python
    import wandb
    from wandb_utils import init_wandb, finish_wandb

    # Initialize WandB before your training loop
    init_wandb(config)

    # Log metrics during training
    wnadb.log({'loss': loss.item()})

    # Finish WandB after your training loop
    finish_wandb(config)
    ```

## Script Overview
The main.py script performs the following steps:
1. Parses command-line arguments using `example_config.get_config()`. This function defines the command-line arguments for the script, including the WandB project name, entity, and API key, and returns a `Namespace`object with the parsed arguments.
2. Initializes WandB using `wandb_utils.init_wandb()`. This function takes a `Namespace` object as input and initializes WandB with the specified project, entity, and API key. It also logs the project settings to the console for debugging purposes.
3. Defines the linear regression model, optimizer, and loss function using PyTorch.
4. Trains the model for 10 epochs using a simple training loop. In each epoch, the script generates random data, performs a forward pass through the model, computes the mean squared error loss, and updates the model parameters using backpropagation.
5. Logs the epoch and loss to WandB using `wandb.log()`. This function takes a dictionary of key-value pairs as input and logs the values to the specified project in WandB.
6. Finishes WandB using `wandb_utils.finish_wandb()`.