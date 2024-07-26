from utils.register import register_dataset
import torch
import torch_geometric
import pandas as pd
from tqdm import tqdm
import shutil
import os

@register_dataset('kinase_inhibitors')
class kinase_inhibitors(torch_geometric.data.InMemoryDataset):
    def __init__(self, root, 
                 create_edge_attr=False, 
                 *args, **kwargs):

        self.root = root
        self.create_edge_attr = create_edge_attr
        self.kwargs = kwargs

        super().__init__(root)
        if self.kwargs['dataset_version'] == 'general_activity_1label':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif self.kwargs['dataset_version'] == 'general_activity_2label':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif self.kwargs['dataset_version'] == 'general_activity_neg1label':
            self.data, self.slices = torch.load(self.processed_paths[2])
        else:
            raise TypeError(f"Unexpected dataset_version {self.kwargs['dataset_version']}")

    @property
    def raw_file_names(self):
        return ['kinase_final.csv']

    @property
    def processed_file_names(self):
        return ['kinases_generally_active_1label.pt',
                'kinase_generally_active_2label.pt',
                'kinase_generally_active_neg1label.pt',
                ]

    def download(self):
        pass
        # shutil.copy("")

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        def generally_active_1label(labels):
            label = 1 if sum(labels) > 0 else 0
            label = torch.tensor([[label]], dtype=torch.float32)
            return label
        def generally_active_2label(labels):
            label = 1 if sum(labels) > 0 else 0
            if label == 1:
                label = torch.tensor([[1,0]], dtype=torch.float32)
            else:
                label = torch.tensor([[0,1]], dtype=torch.float32)
            return label
        def generally_active_neg1label(labels):
            label = 1 if sum(labels) > 0 else -1
            label = torch.tensor([[label]], dtype=torch.float32)
            return label

        data_generally_active_1label = self.df_to_graphs_list(df, generally_active_1label)
        data, slices = self.collate(data_generally_active_1label)
        torch.save((data, slices), self.processed_paths[0])

        data_generally_active_2label = self.df_to_graphs_list(df, generally_active_2label)
        data, slices = self.collate(data_generally_active_2label)
        torch.save((data, slices), self.processed_paths[1])

        data_generally_active_2label = self.df_to_graphs_list(df, generally_active_neg1label)
        data, slices = self.collate(data_generally_active_2label)
        torch.save((data, slices), self.processed_paths[2])
        
    def df_to_graphs_list(self, df, label_func) -> list:
        from rdkit import Chem
        data_list = [None]*len(df)
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Procressing: {label_func.__name__}"):
            smiles = row[0]
            labels = row[1:]
            mol = Chem.MolFromSmiles(smiles)
            
            x = torch.empty([mol.GetNumAtoms(), 5], dtype=torch.float32)
            for i, atom in enumerate(mol.GetAtoms()):
                x[i][0] = atom.GetAtomicNum()
                x[i][1] = atom.GetDegree()
                x[i][2] = atom.GetFormalCharge()
                x[i][3] = atom.GetHybridization()
                x[i][4] = atom.GetIsAromatic()
            
            if self.create_edge_attr:
                edge_indices = []
                edge_attrs = []
                for i, bond in enumerate(mol.GetBonds()):
                    atom1 = bond.GetBeginAtomIdx()
                    atom2 = bond.GetEndAtomIdx()
                    attr = 4 if bond.GetIsAromatic() else int(bond.GetBondTypeAsDouble()) # cast aromatic value (1.5) to int 4                   
                    edge_indices += [[atom1, atom2], [atom2,atom1]]
                    edge_attrs += [attr, attr]
                edge_index = torch.tensor(edge_indices)
                edge_index = edge_index.t().to(torch.long).view(2,-1)
                edge_attr = torch.tensor(edge_attrs, dtype=torch.long).unsqueeze(1)

                if edge_index.numel() > 0:  # Sort indices.
                    perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                    edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]
            else:
                adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
                edge_index, _ = torch_geometric.utils.dense_to_sparse(torch.tensor(adj))
            
            label = label_func(labels)

            data_list[idx] = torch_geometric.data.Data(x=x, 
                                                edge_index=edge_index, 
                                                y=label,
                                                smiles=smiles) 
            if self.create_edge_attr: data_list[idx].edge_attr = edge_attr
        return data_list
       