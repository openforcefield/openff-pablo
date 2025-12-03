from io import StringIO

import pytest

from openff.pablo._pdb import (
    topology_from_pdb,
)
from openff.pablo.ccd._ccdcache import CcdCache
from openff.pablo.residue import ResidueDefinition


def test_load_file_object(cys_pdblines: list[str], tmp_ccd_cache: CcdCache):
    with StringIO(initial_value="\n".join(cys_pdblines)) as f:
        top = topology_from_pdb(f, residue_library=tmp_ccd_cache)
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
        topology_from_pdb(f, residue_library=resdb)

        f.write(f"\n{conect_record}\n")
        f.seek(0)
        with pytest.raises(
            ValueError,
            match="|".join(
                [
                    r"\[H\+:1\] didn't match: found CONECT record that could not be matched with a bond",
                    r"CONECT records without chemical information not supported",
                ],
            ),
        ):
            topology_from_pdb(f, residue_library=resdb)
