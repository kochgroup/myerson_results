# %%
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem as Chem
import math
from collections import defaultdict
import numpy as np

def find_atoms_in_radius(mol, center_atom_idx:int, radius:int, current_radius=0, visited=None):
    if visited is None:
        visited = set()
    if current_radius > radius:
        return []

    visited.add(center_atom_idx)
    atom_idxs_in_radius = [center_atom_idx]

    neighbor_atoms = mol.GetAtomWithIdx(center_atom_idx).GetNeighbors()
    for n in neighbor_atoms:
        if n.GetIdx() not in visited:
            atom_idxs_in_radius.extend(
                find_atoms_in_radius(mol, n.GetIdx(), radius, current_radius+1, visited))
    return list(set(atom_idxs_in_radius))

def find_inner_bonds(mol, inner_atom_idxs: list[int]):
    inner_bond_idxs = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()  
        a2 = bond.GetEndAtomIdx()  
        if a1 in inner_atom_idxs and a2 in inner_atom_idxs:
            inner_bond_idxs.append(bond.GetIdx())
    return inner_bond_idxs

def find_outer_bonds(mol, inner_atom_idxs:list[int], atom_idxs:list[int]):
    outer_bond_idxs = []
    outer_atom_idxs = list(set(atom_idxs).difference(set(inner_atom_idxs)))
    for x in outer_atom_idxs: assert x not in inner_atom_idxs, f"{x} in inner_atom_idxs"
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()  
        a2 = bond.GetEndAtomIdx()  
        one_in_outer = ((a1 in outer_atom_idxs) + (a2 in outer_atom_idxs)) == 1
        one_in_inner = ((a1 in inner_atom_idxs) +  (a2 in inner_atom_idxs)) == 1 
        if one_in_outer and one_in_inner:
            outer_bond_idxs.append(bond.GetIdx())
    return outer_bond_idxs


def find_bonds_in_radius(mol, center_atom_idx:int, radius:int, atom_idxs:list[int]):
    # complicated, so that bonds of 3-rings on the edge are not highlighted
    if radius == 0:
        return []
    inner_atom_idxs = find_atoms_in_radius(mol, center_atom_idx, radius-1)
    bond_idxs = find_inner_bonds(mol, inner_atom_idxs)
    bond_idxs.extend(find_outer_bonds(mol, inner_atom_idxs, atom_idxs))
    return bond_idxs

def get_highlight_for_bit(bit: int, mol, bit_info_map) -> tuple[list[int],list[int]]:
    assert bit in bit_info_map.keys(), f"Bit {bit} not found in bit_info_map"
    atom_hlt = []
    bond_hlt = []
    for center_atom, radius in bit_info_map[bit]:
        atom_idxs = find_atoms_in_radius(mol, center_atom, radius)
        atom_hlt.extend(atom_idxs)
        atom_hlt = list(set(atom_hlt))
        bond_idxs = find_bonds_in_radius(mol, center_atom, radius, atom_idxs)
        bond_hlt.extend(bond_idxs)
        bond_hlt = list(set(bond_hlt))
    return atom_hlt, bond_hlt

def highlight_bit(mol, atom_hlt:list[int], bond_hlt:list[int], legend="",
                   color=(1.,0.5,0.5), size=(500, 500)) -> str:
    highlight_atom_map = {a: [color] for a in atom_hlt}   
    highlight_bond_map = {b: [color] for b in bond_hlt}
    highlight_radii = {}
    highlight_linewidth_multipliers = {}
    d2d = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    d2d.DrawMoleculeWithHighlights(mol,
                                   legend,
                                   highlight_atom_map,
                                   highlight_bond_map,
                                   highlight_radii,
                                   highlight_linewidth_multipliers)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()

def highlight_fingerprint_bits(mol, bits:list[int], bit_info_map, legend:str="") -> str:
    d2d = rdMolDraw2D.MolDraw2DSVG(500, 500)
    d2d.drawOptions().useBWAtomPalette()
    colors = [(1,0.5,0.5),(0,0.8,0.8),(0.8,0.8,0),(0.8,0.0,0.8)]*math.ceil(len(bits)/5)
    highlight_atom_map = defaultdict(list)
    highlight_bond_map = defaultdict(list)
    highlight_radii = {}
    highlight_linewidth_multipliers = {}
    for i, bit in enumerate(bits):
        atom_hlt, bond_hlt = get_highlight_for_bit(bit, mol, bit_info_map)
        for a in atom_hlt:
            highlight_atom_map[a].append(colors[i])
        for b in bond_hlt:
            highlight_bond_map[b].append(colors[i])
    d2d.DrawMoleculeWithHighlights(mol,
                                   legend,
                                   dict(highlight_atom_map),
                                   dict(highlight_bond_map),
                                   highlight_radii,
                                   highlight_linewidth_multipliers)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()

def get_important_shap_bits(shap_values:np.ndarray, num_bits:int) -> list:
    if num_bits==1:
        return [np.argmax(np.abs(shap_values))]
    else:
        return list(np.argsort(np.abs(shap_values))[-1:-num_bits-1:-1])

def find_bit(bit:int, fp_generator, smiles_list):
    for i, smi in enumerate(smiles_list):
        ao = Chem.AdditionalOutput()
        ao.CollectBitInfoMap()
        mol = Chem.MolFromSmiles(smi)
        fp = fp_generator.GetFingerprint(mol, additionalOutput=ao)
        bi = ao.GetBitInfoMap()
        if bit in bi.keys():
            return mol, bi
