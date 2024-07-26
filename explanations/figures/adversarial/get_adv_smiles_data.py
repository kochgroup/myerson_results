# %%
import sys, os
# check for correct working directory 
assert os.getcwd().split('/')[-1] == "adversarial"
import yaml
import pandas as pd
# add higher level directories to the path
path_to_root = os.getcwd()
while path_to_root.split('/')[-1] != 'myerson_results': path_to_root = '/'.join(path_to_root.split('/')[:-1])
sys.path.insert(0,path_to_root)
sys.path.insert(0, os.path.join(path_to_root, "training"))
import explanations.visualizations as vis
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from tqdm import tqdm
import pandas as pd
# %%
data_path = # data not available
all_data = {}
filenames = []

for f in os.listdir(data_path):
    if f.startswith("explanations_val") and not "26" in f:
        filenames.append(f)
        with open(os.path.join(data_path, f), 'r') as file:
            data = yaml.safe_load(file)
        all_data.update(data)
print(filenames)
# %%
tp = 0; tp_keys = []
fp = 0; fp_keys = []
tn = 0; tn_keys = []
fn = 0; fn_keys = []

for k in all_data.keys():
    cfe = all_data[k]['confusion_matrix_entry']
    if cfe == 'tp': tp += 1; tp_keys.append(k)
    elif cfe == 'fp': fp += 1; fp_keys.append(k)
    elif cfe == 'tn': tn += 1; tn_keys.append(k)
    elif cfe == 'fn': fn += 1; fn_keys.append(k)
    else: raise Exception(f"unexpected confusion matrix entry {cfe}")

print(f"Num molecules={len(all_data.keys())}")
print(f"tp={tp}, fp={fp}, tn={tn}, fn={fn}")
print(f"sum={sum([tp, fp, tn, fn])}")

# %%
def methylate(smiles_or_mol:str|Chem.rdchem.Mol, position:int|list[int]):
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    else: 
        mol = smiles_or_mol
    num_atoms = mol.GetNumAtoms()
    if isinstance(position, int):
        methyl = Chem.MolFromSmiles('C')
        mol = Chem.CombineMols(mol, methyl)
        if mol.GetAtomWithIdx(position).GetNumExplicitHs() > 0:
            mol.GetAtomWithIdx(position).SetNumExplicitHs(0)
        romol = Chem.EditableMol(mol)
        romol.AddBond(position, num_atoms, Chem.BondType.SINGLE)
        mol = romol.GetMol()
        return Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    elif isinstance(position, list):
        methyl = [Chem.MolFromSmiles('C') for i in position]
        for p in position:
            if mol.GetAtomWithIdx(p).GetNumExplicitHs() > 0:
                mol.GetAtomWithIdx(p).SetNumExplicitHs(0)
        for m in methyl:
            mol = Chem.CombineMols(mol, m)
        romol = Chem.EditableMol(mol)
        for i,p in enumerate(position):
            romol.AddBond(p, num_atoms+i, Chem.BondType.SINGLE)
        mol = romol.GetMol()
        return Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    else:
        raise TypeError(f"Unexpected argument position {position} of type {type(position)}")

from copy import deepcopy
def _mol_is_valid(mol):
    try:
        mol_copy = deepcopy(mol)
        Chem.SanitizeMol(mol_copy)
        assert Chem.MolFromSmiles(Chem.MolToSmiles(mol_copy)) is not None
        return True
    except:
        return False

def carbonate_position(smiles_or_mol:str|Chem.rdchem.Mol, position:int):
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    else: 
        mol = smiles_or_mol
    mod_atom = mol.GetAtomWithIdx(position)
    if mod_atom.GetAtomicNum() == 6:
        print(f"Warning: replacing carbon at {mod_atom.GetIdx()} with carbon")
    is_aromatic = mod_atom.GetIsAromatic()
    num_explicit_Hs = mod_atom.GetNumExplicitHs()
    mod_atom.SetAtomicNum(6)
    # print(f"checking for validity")
    if not _mol_is_valid(mol) and is_aromatic:
        # print(f"mol was not valid and is aromatic")
        mod_atom.SetIsAromatic(False)
        mod_atom.SetNumExplicitHs(num_explicit_Hs+1)
        mod_atom.SetHybridization(Chem.rdchem.HybridizationType.SP3)
    if not _mol_is_valid(mol):
        raise TypeError(f"not valid and not aromatic, {mod_atom.GetHybridization()=}")
    # print(f"sanitizing")
    Chem.SanitizeMol(mol)
    # print(f"mol(smiles(mol))")
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    assert mol is not None, f"invalid mol before returning"
    # print(f"returning")
    return mol
def carbonate_positions(smiles_or_mol:str|Chem.rdchem.Mol, positions:list[int]):
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    else: 
        mol = smiles_or_mol
    for pos in positions:
        mol = carbonate_position(mol, pos)
    return mol
def get_N_positions(smiles_or_mol:str|Chem.rdchem.Mol):
    N_positions = []
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    else: 
        mol = smiles_or_mol
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 7:
            if len(atom.GetNeighbors()) != 3:
                N_positions.append(atom.GetIdx())
    return N_positions


# %%
def view_mol(smiles_or_mol:str|Chem.rdchem.Mol):
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    else: 
        mol = smiles_or_mol
    for atom in mol.GetAtoms():
        atom.SetProp('atomNote', str(atom.GetIdx()))
    display(Draw.MolToImage(mol)) # type: ignore

# %%
tp_original_smiles = []
tp_adv_smiles = []
for i, k in enumerate(tp_keys):
    smi = all_data[k]['smiles']
    original_smi = all_data[k]['smiles']
    N_positions = get_N_positions(smi)
    c = 0
    while N_positions:
        smi = Chem.MolToSmiles(carbonate_position(smi, N_positions[0]))
        N_positions = get_N_positions(smi)
        c+=1
        if c > 100:
            raise Exception(f"could not replace Ns in {k}, {original_smi}")
    tp_original_smiles.append(original_smi)
    tp_adv_smiles.append(smi)
# %%
# with open('tp_adv_smiles_valset.txt', 'w') as f:
#     for line in tp_adv_smiles:
#         f.write(line+'\n')
df = pd.DataFrame({"tp_keys": tp_keys,
                   "original_smiles": tp_original_smiles,
                   "adv_smiles": tp_adv_smiles,
                   })
df.to_csv('tp_adv_smiles_valset.txt', index=False)
# %%

for i, smi in enumerate(tp_adv_smiles):
    if i < 10:
        print(i)
        view_mol(smi)