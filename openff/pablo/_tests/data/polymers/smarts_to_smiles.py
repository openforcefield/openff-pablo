"""
Converts old-style SMARTS patterns to SMILES appropriate for
`anon_from_smiles_marked_nonleaving`
"""

import json
import re
from pathlib import Path

import rdkit.Chem
from tqdm import tqdm


def get_smarts_atom_degree(atom) -> int | None:
    atom_smarts = atom.GetSmarts()
    assert "," not in atom_smarts
    assert "!" not in atom_smarts
    degrees = re.findall(r"D([0-9]*)[&;\]]", atom_smarts)
    match len(degrees):
        case 0:
            return ValueError("degree not defined on atom")
        case 1:
            return int(degrees[0])
        case _:
            raise ValueError("degree multiply defined on atom")


def get_smarts_atom_charge(atom) -> int:
    atom_smarts = atom.GetSmarts()
    assert "," not in atom_smarts
    assert "!" not in atom_smarts
    formal_charge = 0
    int_found: bool = False
    lone_sign_found: bool = False
    for sign, charge in re.findall(r"([-+])([0-9]*)", atom_smarts):
        match sign, charge:
            case "+", "":
                formal_charge += 1
                lone_sign_found = True
            case "-", "":
                formal_charge -= 1
                lone_sign_found = True
            case "+", charge:
                formal_charge = int(charge)
                int_found = True
            case "-", charge:
                formal_charge = -int(charge)
                int_found = True
            case _, _:
                assert False, "unreachable"

    if (
        (atom_smarts.count("+") + atom_smarts.count("-") == 0)
        or (atom_smarts.count("+") != 0 and atom_smarts.count("-") != 0)
        or (int_found and lone_sign_found)
    ):
        raise ValueError(
            f"ambiguous charge on atom: {atom_smarts}",
        )
    return formal_charge


def smarts_to_smiles(smarts: str) -> str:
    mol = rdkit.Chem.MolFromSmarts(smarts)
    map_to_degree = {}
    new_to_old_map = {}
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() == 0:
            # This was a * in the SMARTS, so we'll make it hydrogen and make
            # sure it's not mapped (it's a leaving atom)
            atom.SetAtomMapNum(0)
            atom.SetAtomicNum(1)
        else:
            degree = get_smarts_atom_degree(atom)
            charge = get_smarts_atom_charge(atom)
            old_map = atom.GetAtomMapNum()
            new_map = i + 1

            atom.SetAtomMapNum(new_map)
            atom.SetFormalCharge(charge)
            map_to_degree[new_map] = degree
            new_to_old_map[new_map] = old_map
    smiles = rdkit.Chem.MolToSmiles(mol)

    # Check that all degrees are correct
    mol_from_smiles = rdkit.Chem.AddHs(rdkit.Chem.MolFromSmiles(smiles))
    for atom in mol_from_smiles.GetAtoms():
        map = atom.GetAtomMapNum()
        degree = len(atom.GetNeighbors())
        if map != 0:
            assert degree == map_to_degree[map], (
                f"expected {map_to_degree[map]} got {degree} for atom map {new_to_old_map[map]}>>>{map}"
            )

    return smiles


def dbg(obj, msg=""):
    if msg:
        print(msg, obj)
    else:
        print(obj)
    return obj


def main():
    smarts_monomers_files = [
        file
        for dir in Path(__file__).parent.iterdir()
        if dir.is_dir()
        for file in dir.iterdir()
        if file.name.endswith(".monomers.json")
    ]
    progressbar = tqdm(smarts_monomers_files)
    for smarts_file in tqdm(smarts_monomers_files):
        progressbar.write(str(smarts_file))
        smiles_file = smarts_file.with_suffix(".smiles.json")
        smarts_data: dict[str, list[str]] = json.loads(smarts_file.read_text())

        assert type(smarts_data) is dict
        assert all(type(name) is str for name in smarts_data)
        assert all(type(value) is list for value in smarts_data.values())
        assert all(
            all(type(elem) is str for elem in value) for value in smarts_data.values()
        )

        smiles_data = {
            key: [smarts_to_smiles(elem) for elem in value]
            for key, value in smarts_data.items()
        }

        smiles_file.write_text(json.dumps(smiles_data))


if __name__ == "__main__":
    main()
