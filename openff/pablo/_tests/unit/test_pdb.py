from io import StringIO

import pytest
from openff.toolkit import Molecule

from openff.pablo._pdb import (
    _add_to_molecule,
    topology_from_pdb,
)
from openff.pablo._pdb_data import PdbData, ResidueMatch
from openff.pablo.residue import ResidueDefinition


def test_add_to_molecule_when_empty(cys_data: PdbData, cys_match: ResidueMatch):
    this_molecule = Molecule()
    molecules = [this_molecule]
    mol_out = _add_to_molecule(
        data=cys_data,
        molecules=molecules,
        this_molecule=this_molecule,
        residue_match=cys_match,
        use_canonical_names=False,
    )
    assert mol_out is this_molecule
    matched, _ = Molecule.are_isomorphic(
        mol_out,
        cys_match.residue_definition.to_openff_molecule(),
        atom_stereochemistry_matching=False,
    )
    assert matched


def test_load_file_object(cys_pdblines: list[str]):
    with StringIO(initial_value="\n".join(cys_pdblines)) as f:
        top = topology_from_pdb(f)
    assert top.n_atoms == 14


def test_conect_to_vsite():
    atom_records = [
        "ATOM      1  H   HPS     1      -0.664  -1.578  -0.633  1.00  0.00           H1+",
        "ATOM      2  EP  HPS     1      -0.105  -0.229  -0.633  1.00  0.00              ",
    ]
    conect_record = "CONECT    1    2"

    resdb = {
        "HPS": [
            ResidueDefinition.from_smiles(
                "[H+:1]",
                atom_names={1: "H"},
                virtual_sites=["EP"],
                residue_name="HPS",
            ),
        ],
    }

    with StringIO("\n".join(atom_records)) as f:
        topology_from_pdb(f, residue_database=resdb)

        f.write(f"\n{conect_record}\n")
        f.seek(0)
        with pytest.raises(
            ValueError,
            match="|".join(
                [
                    r"\[H\+:1\] failed to match: found CONECT record that could not be matched with a bond",
                    r"CONECT records without chemical information not supported",
                ],
            ),
        ):
            topology_from_pdb(f, residue_database=resdb)
