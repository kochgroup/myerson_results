import requests
from codecs import decode

def smiles_to_atom_count(smiles):
    # only works for kekulized SMILES
    non_atomic_symbols = [".","-","+","=","#","$",":","/","\\","@","%","[","]","(",")"]
    explicit_H_symbol = ["H"]
    numbers = [str(x) for x in range(10)]
    for symbol in non_atomic_symbols+explicit_H_symbol+numbers:
        smiles = smiles.replace(symbol, "")
    elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
        'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
        'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er',
        'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
        'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
        'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
        'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
    atom_counter = 0
    for el in [x for x in elements if len(x) > 1]: # two-character elements first
        atom_counter += smiles.count(el)
        smiles = smiles.replace(el, "")
    for el in [x for x in elements if len(x) == 1]:
        atom_counter += smiles.count(el)
        smiles = smiles.replace(el, "")
    assert smiles == "", f"SMILES still has symbols: {smiles}"
    return atom_counter

def get_smiles(how_many, num_atoms=False, randomize=False):
    """Get canonical/kekulized SMILES strings from the PubChem database.

    Args:
        how_many (int): How many SMILES to return.
        num_atoms (bool, int, optional): How many atoms the SMILES should have. 
            If False, always start at compound id 1. Defaults to False.
        randomize (bool, optional): Whether to query the databse randomly. 
            Defaults to False.

    Returns:
        list: The list of SMILES.
    """
    # randomize not yet included
    smiles_list = []
    if randomize:
        import random
        cid = random.randint(1, 160000000)
    else:
        cid = 1
    while len(smiles_list) < how_many:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSmiles/txt"
        req = requests.get(url)
        raw = decode(req.content)
        smiles = raw[:-1] # remove trailing linebreak
        if num_atoms:
            if smiles_to_atom_count(smiles) == num_atoms:
                smiles_list.append(smiles)
        else:
            smiles_list.append(smiles)
        cid += 1 
    return smiles_list
