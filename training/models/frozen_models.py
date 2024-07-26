from utils.register import register_model
import torch
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_add_pool, global_max_pool 
from torch.nn import functional as F
from torch.nn import Linear

@register_model('gat_frozen3conv1fc')
class GATConvModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=256, pool='mean', freeze=True, *args, **kwargs):
        super().__init__()
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unexpected pooling function: {pool}")
        self.conv1 = GATConv(dim_in, dim_hidden)
        self.conv2 = GATConv(dim_hidden, dim_hidden)
        self.conv3 = GATConv(dim_hidden, dim_hidden)
        self.fc1 = Linear(dim_hidden, dim_out)

        if freeze:
            c = 0
            for child in self.children():
                c+=1
                if c < 4:
                    for param in child.parameters():
                        param.requires_grad=False

    def forward(self, x, edge_index, batch, *args, **kwargs):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.pool(x, batch)
        x = self.fc1(x)
        
        return x


@register_model('gat_frozen3conv3fc')
class GATConvModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=256, pool='mean', freeze=True, *args, **kwargs):
        super().__init__()
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unexpected pooling function: {pool}")
        self.conv1 = GATConv(dim_in, dim_hidden)
        self.conv2 = GATConv(dim_hidden, dim_hidden)
        self.conv3 = GATConv(dim_hidden, dim_hidden)
        self.fc1 = Linear(dim_hidden, dim_hidden)
        self.fc2 = Linear(dim_hidden, dim_hidden)
        self.fc3 = Linear(dim_hidden, dim_out)

        if freeze:
            c = 0
            for child in self.children():
                c+=1
                if c < 4:
                    for param in child.parameters():
                        param.requires_grad=False

    def forward(self, x, edge_index, batch, *args, **kwargs):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.pool(x, batch)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x