from utils.register import register_dataset
import torch
import torch_geometric
import pandas as pd
from tqdm import tqdm
import shutil
import os

@register_dataset('mw_smallzinc')
class mw_smallzinc(torch_geometric.data.InMemoryDataset):
    def __init__(self, root, 
                 create_edge_attr=False, 
                 *args, **kwargs):

        self.root = root
        self.create_edge_attr = create_edge_attr

        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['less_11_atom_MW.csv']

    @property
    def processed_file_names(self):
        return ['mw_smallzinc.pt']

    def download(self):
        shutil.copy("../data/less_11_atom_MW.csv", self.raw_dir)

    def process(self):
        from rdkit import Chem
        df = pd.read_csv(self.raw_paths[0], header=None)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        data_list = [None]*len(df)
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            zinc_id = row[0]
            smiles = row[1]
            label = row[2]
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
            
            label = torch.tensor([[label]], dtype=torch.float32)

            data_list[idx] = torch_geometric.data.Data(x=x, 
                                                  edge_index=edge_index, 
                                                  y=label,
                                                  smiles=smiles,
                                                  zinc_id=zinc_id)
            if self.create_edge_attr: data_list[idx].edge_attr = edge_attr

            

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        