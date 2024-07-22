import time  # Importing the time module for performance tracking
import logging  # Importing the logging module for logging information
from util import create_parser, set_seed, logger_setup  # Importing utility functions
from data_loading import get_data  # Importing the data loading function
from training import train_gnn  # Importing the training function for the GNN
from inference import infer_gnn  # Importing the inference function for the GNN
import json  # Importing the JSON module to handle JSON files


def main():
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()

    # Load data configuration from a JSON file
    with open('data_config.json', 'r') as config_file:
        data_config = json.load(config_file)

    # Setup logging configuration
    logger_setup()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Log the start of data retrieval
    logging.info("Retrieving data")
    t1 = time.perf_counter()  # Start timing the data retrieval process

    # Retrieve the data
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(
        args, data_config)

    t2 = time.perf_counter()  # End timing the data retrieval process
    # Log the time taken for data retrieval
    logging.info(f"Retrieved data in {t2-t1:.2f}s")

    if not args.inference:
        # If not in inference mode, run training
        logging.info(f"Running Training")
        train_gnn(tr_data, val_data, te_data, tr_inds,
                  val_inds, te_inds, args, data_config)
    else:
        # If in inference mode, run inference
        logging.info(f"Running Inference")
        infer_gnn(tr_data, val_data, te_data, tr_inds,
                  val_inds, te_inds, args, data_config)


# Run the main function if this script is executed
if __name__ == "__main__":
    main()
