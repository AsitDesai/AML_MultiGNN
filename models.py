import torch.nn as nn
from torch_geometric.nn import GINEConv, BatchNorm, Linear, GATConv, PNAConv, RGCNConv
import torch.nn.functional as F
import torch
import logging

class GINe(torch.nn.Module):
    """
    Graph Isomorphism Network (GIN) with edge updates.

    Attributes:
    - num_features: Number of input node features.
    - num_gnn_layers: Number of GNN layers.
    - n_classes: Number of output classes.
    - n_hidden: Number of hidden units in the layers.
    - edge_updates: Whether to use edge updates.
    - edge_dim: Dimensionality of edge features.
    - dropout: Dropout probability for the hidden layers.
    - final_dropout: Dropout probability for the final classification layer.
    """
    def __init__(self, num_features, num_gnn_layers, n_classes=2, 
                n_hidden=100, edge_updates=False, residual=True, 
                edge_dim=None, dropout=0.0, final_dropout=0.5):
        super().__init__()
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout

        # Node and edge embedding layers
        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        # Initialize lists for GNN layers, edge update layers, and batch normalization
        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(self.num_gnn_layers):
            # Create a GINEConv layer
            conv = GINEConv(nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden), 
                nn.ReLU(), 
                nn.Linear(self.n_hidden, self.n_hidden)
                ), edge_dim=self.n_hidden)
            # Create edge update layers if required
            if self.edge_updates:
                self.emlps.append(nn.Sequential(
                    nn.Linear(3 * self.n_hidden, self.n_hidden),
                    nn.ReLU(),
                    nn.Linear(self.n_hidden, self.n_hidden),
                ))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        # Define a multi-layer perceptron (MLP) for classification
        self.mlp = nn.Sequential(
            Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),
            Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
            Linear(25, n_classes)
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass for the GINe model.

        Args:
        - x: Node feature matrix.
        - edge_index: Edge index tensor.
        - edge_attr: Edge feature matrix.

        Returns:
        - out: Model output after applying MLP.
        """
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        out = x
        
        return self.mlp(out)

class GATe(torch.nn.Module):
    """
    Graph Attention Network (GAT) with edge updates.

    Attributes:
    - num_features: Number of input node features.
    - num_gnn_layers: Number of GNN layers.
    - n_classes: Number of output classes.
    - n_hidden: Number of hidden units in the layers.
    - n_heads: Number of attention heads.
    - edge_updates: Whether to use edge updates.
    - edge_dim: Dimensionality of edge features.
    - dropout: Dropout probability for the hidden layers.
    - final_dropout: Dropout probability for the final classification layer.
    """
    def __init__(self, num_features, num_gnn_layers, n_classes=2, n_hidden=100, n_heads=4, edge_updates=False, edge_dim=None, dropout=0.0, final_dropout=0.5):
        super().__init__()
        # Adjust hidden units based on number of heads
        tmp_out = n_hidden // n_heads
        n_hidden = tmp_out * n_heads

        self.n_hidden = n_hidden
        self.n_heads = n_heads
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.dropout = dropout
        self.final_dropout = final_dropout
        
        # Node and edge embedding layers
        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)
        
        # Initialize lists for GAT layers, edge update layers, and batch normalization
        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(self.num_gnn_layers):
            # Create a GATConv layer
            conv = GATConv(self.n_hidden, tmp_out, self.n_heads, concat=True, dropout=self.dropout, add_self_loops=True, edge_dim=self.n_hidden)
            if self.edge_updates:
                self.emlps.append(nn.Sequential(
                    nn.Linear(3 * self.n_hidden, self.n_hidden),
                    nn.ReLU(),
                    nn.Linear(self.n_hidden, self.n_hidden),
                ))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))
                
        # Define a multi-layer perceptron (MLP) for classification
        self.mlp = nn.Sequential(
            Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),
            Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
            Linear(25, n_classes)
        )
            
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass for the GATe model.

        Args:
        - x: Node feature matrix.
        - edge_index: Edge index tensor.
        - edge_attr: Edge feature matrix.

        Returns:
        - out: Model output after applying MLP.
        """
        src, dst = edge_index
        
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        
        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2
                    
        logging.debug(f"x.shape = {x.shape}, x[edge_index.T].shape = {x[edge_index.T].shape}")
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        logging.debug(f"x.shape = {x.shape}")
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        logging.debug(f"x.shape = {x.shape}")
        out = x

        return self.mlp(out)

class PNA(torch.nn.Module):
    """
    Principal Neighbourhood Aggregation (PNA) with edge updates.

    Attributes:
    - num_features: Number of input node features.
    - num_gnn_layers: Number of GNN layers.
    - n_classes: Number of output classes.
    - n_hidden: Number of hidden units in the layers.
    - edge_updates: Whether to use edge updates.
    - edge_dim: Dimensionality of edge features.
    - dropout: Dropout probability for the hidden layers.
    - final_dropout: Dropout probability for the final classification layer.
    - deg: Degree information for nodes.
    """
    def __init__(self, num_features, num_gnn_layers, n_classes=2, 
                n_hidden=100, edge_updates=True,
                edge_dim=None, dropout=0.0, final_dropout=0.5, deg=None):
        super().__init__()
        n_hidden = int((n_hidden // 5) * 5)  # Ensure that hidden units are divisible by 5
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout

        # Aggregators and scalers used in PNA
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        # Node and edge embedding layers
        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        # Initialize lists for PNA layers, edge update layers, and batch normalization
        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            # Create a PNAConv layer
            conv = PNAConv(in_channels=n_hidden, out_channels=n_hidden,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=n_hidden, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            if self.edge_updates:
                self.emlps.append(nn.Sequential(
                    nn.Linear(3 * self.n_hidden, self.n_hidden),
                    nn.ReLU(),
                    nn.Linear(self.n_hidden, self.n_hidden),
                ))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        # Define a multi-layer perceptron (MLP) for classification
        self.mlp = nn.Sequential(
            Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),
            Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
            Linear(25, n_classes)
        )

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass for the PNA model.

        Args:
        - x: Node feature matrix.
        - edge_index: Edge index tensor.
        - edge_attr: Edge feature matrix.

        Returns:
        - out: Model output after applying MLP.
        """
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        out = x
        return self.mlp(out)

class RGCN(nn.Module):
    """
    Relational Graph Convolutional Network (RGCN) with optional edge updates.

    Attributes:
    - num_features: Number of input node features.
    - edge_dim: Dimensionality of edge features.
    - num_relations: Number of relation types.
    - num_gnn_layers: Number of GNN layers.
    - n_classes: Number of output classes.
    - n_hidden: Number of hidden units in the layers.
    - edge_update: Whether to use edge updates.
    - residual: Whether to use residual connections.
    - dropout: Dropout probability for the hidden layers.
    - final_dropout: Dropout probability for the final classification layer.
    - n_bases: Number of basis relations.
    """
    def __init__(self, num_features, edge_dim, num_relations, num_gnn_layers, n_classes=2, 
                n_hidden=100, edge_update=False,
                residual=True,
                dropout=0.0, final_dropout=0.5, n_bases=-1):
        super(RGCN, self).__init__()

        self.num_features = num_features
        self.num_gnn_layers = num_gnn_layers
        self.n_hidden = n_hidden
        self.residual = residual
        self.dropout = dropout
        self.final_dropout = final_dropout
        self.n_classes = n_classes
        self.edge_update = edge_update
        self.num_relations = num_relations
        self.n_bases = n_bases

        # Node and edge embedding layers
        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        # Initialize lists for RGCN layers, edge update layers, and batch normalization
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.mlp = nn.ModuleList()

        if self.edge_update:
            self.emlps = nn.ModuleList()
            self.emlps.append(nn.Sequential(
                nn.Linear(3 * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))
        
        for _ in range(self.num_gnn_layers):
            # Create an RGCNConv layer
            conv = RGCNConv(self.n_hidden, self.n_hidden, num_relations, num_bases=self.n_bases)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(self.n_hidden))

            if self.edge_update:
                self.emlps.append(nn.Sequential(
                    nn.Linear(3 * self.n_hidden, self.n_hidden),
                    nn.ReLU(),
                    nn.Linear(self.n_hidden, self.n_hidden),
                ))

        # Define a multi-layer perceptron (MLP) for classification
        self.mlp = nn.Sequential(
            Linear(n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),
            Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
            Linear(25, n_classes)
        )

    def reset_parameters(self):
        """
        Reset parameters of the model layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, RGCNConv):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass for the RGCN model.

        Args:
        - x: Node feature matrix.
        - edge_index: Edge index tensor.
        - edge_attr: Edge feature matrix.

        Returns:
        - out: Model output after applying MLP.
        """
        edge_type = edge_attr[:, -1].long()
        src, dst = edge_index

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x =  (x + F.relu(self.bns[i](self.convs[i](x, edge_index, edge_type)))) / 2
            if self.edge_update:
                edge_attr = (edge_attr + F.relu(self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)))) / 2
        
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        x = self.mlp(x)
        out = x

        return x
