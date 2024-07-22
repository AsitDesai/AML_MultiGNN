import torch  # PyTorch for tensor operations
import tqdm  # Progress bar library
from sklearn.metrics import f1_score  # Metric for evaluating model performance
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, evaluate_hetero, save_model, load_model  # Utility functions for training
from models import GINe, PNA, GATe, RGCN  # Importing various GNN models
from torch_geometric.data import Data, HeteroData  # PyTorch Geometric data structures
from torch_geometric.nn import to_hetero, summary  # PyTorch Geometric utilities
from torch_geometric.utils import degree  # Utility for calculating degree of nodes
import wandb  # Logging and monitoring tool
import logging  # Logging library

def train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    """
    Train a homogeneous GNN model.

    Args:
        tr_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        te_loader (DataLoader): Test data loader.
        tr_inds (Tensor): Training indices.
        val_inds (Tensor): Validation indices.
        te_inds (Tensor): Test indices.
        model (nn.Module): GNN model.
        optimizer (Optimizer): Optimizer for model parameters.
        loss_fn (nn.Module): Loss function.
        args (Namespace): Command line arguments.
        config (dict): Configuration dictionary.
        device (torch.device): Device to run the model on.
        val_data (Data): Validation data.
        te_data (Data): Test data.
        data_config (dict): Data configuration dictionary.

    Returns:
        nn.Module: Trained GNN model.
    """
    best_val_f1 = 0  # Best validation F1 score
    for epoch in range(config.epochs):  # Training loop
        total_loss = total_examples = 0  # Initialize loss and example counters
        preds = []  # List to store predictions
        ground_truths = []  # List to store ground truths
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):  # Iterate over batches
            optimizer.zero_grad()  # Reset gradients
            # Select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch.input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

            # Remove the unique edge id from the edge features
            batch.edge_attr = batch.edge_attr[:, 1:]

            batch.to(device)  # Move batch to device
            out = model(batch.x, batch.edge_index, batch.edge_attr)  # Forward pass
            pred = out[mask]  # Predictions for the masked edges
            ground_truth = batch.y[mask]  # Ground truth for the masked edges
            preds.append(pred.argmax(dim=-1))  # Store predictions
            ground_truths.append(ground_truth)  # Store ground truths
            loss = loss_fn(pred, ground_truth)  # Compute loss

            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters

            total_loss += float(loss) * pred.numel()  # Accumulate loss
            total_examples += pred.numel()  # Accumulate number of examples

        # Calculate F1 score for training data
        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        f1 = f1_score(ground_truth, pred)
        wandb.log({"f1/train": f1}, step=epoch)  # Log training F1 score
        logging.info(f'Train F1: {f1:.4f}')

        # Evaluate the model
        val_f1 = evaluate_homo(val_loader, val_inds, model, val_data, device, args)
        te_f1 = evaluate_homo(te_loader, te_inds, model, te_data, device, args)

        wandb.log({"f1/validation": val_f1}, step=epoch)  # Log validation F1 score
        wandb.log({"f1/test": te_f1}, step=epoch)  # Log test F1 score
        logging.info(f'Validation F1: {val_f1:.4f}')
        logging.info(f'Test F1: {te_f1:.4f}')

        # Save the best model based on validation F1 score
        if epoch == 0:
            wandb.log({"best_test_f1": te_f1}, step=epoch)
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wandb.log({"best_test_f1": te_f1}, step=epoch)
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)
    
    return model  # Return the trained model

def train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config):
    """
    Train a heterogeneous GNN model.

    Args:
        tr_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        te_loader (DataLoader): Test data loader.
        tr_inds (Tensor): Training indices.
        val_inds (Tensor): Validation indices.
        te_inds (Tensor): Test indices.
        model (nn.Module): GNN model.
        optimizer (Optimizer): Optimizer for model parameters.
        loss_fn (nn.Module): Loss function.
        args (Namespace): Command line arguments.
        config (dict): Configuration dictionary.
        device (torch.device): Device to run the model on.
        val_data (HeteroData): Validation data.
        te_data (HeteroData): Test data.
        data_config (dict): Data configuration dictionary.

    Returns:
        nn.Module: Trained GNN model.
    """
    best_val_f1 = 0  # Best validation F1 score
    for epoch in range(config.epochs):  # Training loop
        total_loss = total_examples = 0  # Initialize loss and example counters
        preds = []  # List to store predictions
        ground_truths = []  # List to store ground truths
        for batch in tqdm.tqdm(tr_loader, disable=not args.tqdm):  # Iterate over batches
            optimizer.zero_grad()  # Reset gradients
            # Select the seed edges from which the batch was created
            inds = tr_inds.detach().cpu()
            batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
            batch_edge_ids = tr_loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
            mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)
            
            # Remove the unique edge id from the edge features
            batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
            batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]

            batch.to(device)  # Move batch to device
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)  # Forward pass
            out = out[('node', 'to', 'node')]  # Extract output for specific edge type
            pred = out[mask]  # Predictions for the masked edges
            ground_truth = batch['node', 'to', 'node'].y[mask]  # Ground truth for the masked edges
            preds.append(pred.argmax(dim=-1))  # Store predictions
            ground_truths.append(batch['node', 'to', 'node'].y[mask])  # Store ground truths
            loss = loss_fn(pred, ground_truth)  # Compute loss

            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters

            total_loss += float(loss) * pred.numel()  # Accumulate loss
            total_examples += pred.numel()  # Accumulate number of examples
            
        # Calculate F1 score for training data
        pred = torch.cat(preds, dim=0).detach().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        f1 = f1_score(ground_truth, pred)
        wandb.log({"f1/train": f1}, step=epoch)  # Log training F1 score
        logging.info(f'Train F1: {f1:.4f}')

        # Evaluate the model
        val_f1 = evaluate_hetero(val_loader, val_inds, model, val_data, device, args)
        te_f1 = evaluate_hetero(te_loader, te_inds, model, te_data, device, args)

        wandb.log({"f1/validation": val_f1}, step=epoch)  # Log validation F1 score
        wandb.log({"f1/test": te_f1}, step=epoch)  # Log test F1 score
        logging.info(f'Validation F1: {val_f1:.4f}')
        logging.info(f'Test F1: {te_f1:.4f}')

        # Save the best model based on validation F1 score
        if epoch == 0:
            wandb.log({"best_test_f1": te_f1}, step=epoch)
        elif val_f1 > best_val_f1:
            best_val_f1 = val_f1
            wandb.log({"best_test_f1": te_f1}, step=epoch)
            if args.save_model:
                save_model(model, optimizer, epoch, args, data_config)
        
    return model  # Return the trained model

def get_model(sample_batch, config, args):
    """
    Get the GNN model based on the provided configuration.

    Args:
        sample_batch (Data or HeteroData): Sample batch of data.
        config (dict): Configuration dictionary.
        args (Namespace): Command line arguments.

    Returns:
        nn.Module: GNN model.
    """
    n_feats = sample_batch.x.shape[1] if not isinstance(sample_batch, HeteroData) else sample_batch['node'].x.shape[1]
    e_dim = (sample_batch.edge_attr.shape[1] - 1) if not isinstance(sample_batch, HeteroData) else (sample_batch['node', 'to', 'node'].edge_attr.shape[1] - 1)

    # Instantiate model based on the selected type
    if args.model == "gin":
        model = GINe(
            num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
            n_hidden=round(config.n_hidden), residual=False, edge_updates=args.emlps, edge_dim=e_dim, 
            dropout=config.dropout, final_dropout=config.final_dropout
        )
    elif args.model == "gat":
        model = GATe(
            num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
            n_hidden=round(config.n_hidden), n_heads=round(config.n_heads), 
            edge_updates=args.emlps, edge_dim=e_dim,
            dropout=config.dropout, final_dropout=config.final_dropout
        )
    elif args.model == "pna":
        if not isinstance(sample_batch, HeteroData):
            d = degree(sample_batch.edge_index[1], dtype=torch.long)
        else:
            index = torch.cat((sample_batch['node', 'to', 'node'].edge_index[1], sample_batch['node', 'rev_to', 'node'].edge_index[1]), 0)
            d = degree(index, dtype=torch.long)
        deg = torch.bincount(d, minlength=1)
        model = PNA(
            num_features=n_feats, num_gnn_layers=config.n_gnn_layers, n_classes=2,
            n_hidden=round(config.n_hidden), edge_updates=args.emlps, edge_dim=e_dim,
            dropout=config.dropout, deg=deg, final_dropout=config.final_dropout
        )
    elif args.model == "rgcn":
        model = RGCN(
            num_features=n_feats, edge_dim=e_dim, num_relations=8, num_gnn_layers=round(config.n_gnn_layers),
            n_classes=2, n_hidden=round(config.n_hidden),
            edge_update=args.emlps, dropout=config.dropout, final_dropout=config.final_dropout, n_bases=None
        )
    
    return model  # Return the instantiated model

def train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
    """
    Train a GNN model.

    Args:
        tr_data (Data or HeteroData): Training data.
        val_data (Data or HeteroData): Validation data.
        te_data (Data or HeteroData): Test data.
        tr_inds (Tensor): Training indices.
        val_inds (Tensor): Validation indices.
        te_inds (Tensor): Test indices.
        args (Namespace): Command line arguments.
        data_config (dict): Data configuration dictionary.

    Returns:
        nn.Module: Trained GNN model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Set device

    # Initialize WandB for logging and monitoring
    wandb.init(
        mode="disabled" if args.testing else "online",
        project="your_proj_name",  # Replace with your WandB project name

        config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "num_neighbors": args.num_neighs,
            "lr": extract_param("lr", args),
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": "ce",
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": extract_param("w_ce2", args),
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "n_heads": extract_param("n_heads", args) if args.model == 'gat' else None
        }
    )

    config = wandb.config  # Set configuration

    # Set the transform if ego ids should be used
    transform = AddEgoIds() if args.ego else None

    # Add unique ids to later find the seed edges
    add_arange_ids([tr_data, val_data, te_data])

    # Get data loaders
    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

    # Get the model
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)

    # Convert to heterogeneous model if necessary
    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr='mean')
    
    # Load or initialize model
    if args.finetune:
        model, optimizer = load_model(model, device, args, config, data_config)
    else:
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    # Prepare sample batch for logging
    sample_batch.to(device)
    sample_x = sample_batch.x if not isinstance(sample_batch, HeteroData) else sample_batch.x_dict
    sample_edge_index = sample_batch.edge_index if not isinstance(sample_batch, HeteroData) else sample_batch.edge_index_dict
    if isinstance(sample_batch, HeteroData):
        sample_batch['node', 'to', 'node'].edge_attr = sample_batch['node', 'to', 'node'].edge_attr[:, 1:]
        sample_batch['node', 'rev_to', 'node'].edge_attr = sample_batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
    else:
        sample_batch.edge_attr = sample_batch.edge_attr[:, 1:]
    sample_edge_attr = sample_batch.edge_attr if not isinstance(sample_batch, HeteroData) else sample_batch.edge_attr_dict
    logging.info(summary(model, sample_x, sample_edge_index, sample_edge_attr))
    
    # Define loss function
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([config.w_ce1, config.w_ce2]).to(device))

    # Train the model
    if args.reverse_mp:
        model = train_hetero(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
    else:
        model = train_homo(tr_loader, val_loader, te_loader, tr_inds, val_inds, te_inds, model, optimizer, loss_fn, args, config, device, val_data, te_data, data_config)
    
    wandb.finish()  # Finish WandB logging

    return model  # Return the trained model
