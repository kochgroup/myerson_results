from utils.register import register_model
import torch
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_add_pool, global_max_pool 
from torch.nn import functional as F
from torch.nn import Linear

@register_model('gat_3conv1fc_softmax')
class GATConvBinClass(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=256, pool='mean', *args, **kwargs):
        super().__init__()
        dim_hidden = 256 if dim_hidden is None else dim_hidden
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unexpected pooling function: {pool}")
        self.conv1 = GATConv(dim_in, dim_hidden)
        self.conv2 = GATConv(dim_hidden, dim_hidden)
        self.conv3 = GATConv(dim_hidden, dim_hidden)
        self.fc1 = Linear(dim_hidden, dim_out)

    def forward(self, x, edge_index, batch, *args, **kwargs):
       
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.pool(x, batch)
        x = self.fc1(x)
        x = F.softmax(x)
        
        return x
