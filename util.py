import argparse
import numpy as np
import torch
import random
import logging
import os
import sys

def logger_setup():
    """
    Setup logging for the application.
    
    - Creates a 'logs' directory if it does not exist.
    - Configures logging to output messages to both a log file and the console.
    """
    log_directory = "logs"
    # Create the 'logs' directory if it does not exist
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_directory, "logs.log")),  # Log to local file
            logging.StreamHandler(sys.stdout)  # Also log to console
        ]
    )

def create_parser():
    """
    Create argument parser for command-line arguments.

    Returns:
    - parser (argparse.ArgumentParser): Configured argument parser.
    """
    parser = argparse.ArgumentParser()

    # Add arguments for adaptations and features
    parser.add_argument("--emlps", action='store_true', help="Use EMLPS (Extended Message Passing Layers) in GNN training")
    parser.add_argument("--reverse_mp", action='store_true', help="Use reverse message passing in GNN training")
    parser.add_argument("--ports", action='store_true', help="Use port numberings in GNN training")
    parser.add_argument("--tds", action='store_true', help="Use time deltas (time between subsequent transactions) in GNN training")
    parser.add_argument("--ego", action='store_true', help="Use ego IDs in GNN training")

    # Add model training parameters
    parser.add_argument("--batch_size", default=8192, type=int, help="Batch size for GNN training")
    parser.add_argument("--n_epochs", default=100, type=int, help="Number of epochs for GNN training")
    parser.add_argument('--num_neighs', nargs='+', default=[100,100], help='Number of neighbors to sample at each hop (descending).')

    # Add miscellaneous options
    parser.add_argument("--seed", default=1, type=int, help="Random seed for reproducibility")
    parser.add_argument("--tqdm", action='store_true', help="Use tqdm for progress logging (when running interactively in terminal)")
    parser.add_argument("--data", default=None, type=str, help="AML dataset to use (small or medium).", required=True)
    parser.add_argument("--model", default=None, type=str, help="Model architecture to use (e.g., gin, gat, rgcn, pna)", required=True)
    parser.add_argument("--testing", action='store_true', help="Disable W&B logging while running in 'testing' mode.")
    parser.add_argument("--save_model", action='store_true', help="Save the best model.")
    parser.add_argument("--unique_name", action='store_true', help="Unique name under which the model will be stored.")
    parser.add_argument("--finetune", action='store_true', help="Fine-tune an existing model. 'unique_name' should point to a pre-trained model.")
    parser.add_argument("--inference", action='store_true', help="Load a trained model for inference only. 'unique_name' should point to the trained model.")

    return parser

def set_seed(seed: int = 0) -> None:
    """
    Set random seed for reproducibility.

    Args:
    - seed (int): Seed value for random number generators.
    """
    # Set seed for various random number generators
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Configure CuDNN backend for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set as {seed}")
