from rdkit import Chem
import torch_geometric
import torch


def GraphFromSmiles(smiles, label=None):
    mol = Chem.MolFromSmiles(smiles)

    x = torch.empty([mol.GetNumAtoms(), 5], dtype=torch.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        x[i][0] = atom.GetAtomicNum()
        x[i][1] = atom.GetDegree()
        x[i][2] = atom.GetFormalCharge()
        x[i][3] = atom.GetHybridization()
        x[i][4] = atom.GetIsAromatic()

    adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    edge_index, _ = torch_geometric.utils.dense_to_sparse(torch.tensor(adj))
                
    if label:
        label = torch.tensor([[label]], dtype=torch.float32)

    graph = torch_geometric.data.Data(x=x, 
                                    edge_index=edge_index, 
                                    y=label,
                                    smiles=smiles)
    return graph 