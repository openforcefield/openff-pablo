import json
from collections.abc import Collection, Mapping
from pathlib import Path

import pytest
from openff.toolkit import Topology

from openff.pablo._pdb import topology_from_pdb
from openff.pablo._tests.utils import get_test_data_path
from openff.pablo._utils import sort_tuple
from openff.pablo.ccd._ccdcache import CcdCache
from openff.pablo.residue import ResidueDefinition

FAST_PDBS: list[tuple[str, str, list[ResidueDefinition]]] = [
    (
        "prepared_pdbs/2MUM_neutralized.pdb",
        "prepared_pdbs/2MUM_neutralized.json",
        [],
    ),
    (
        "prepared_pdbs/2MUM_dryrun.pdb",
        "prepared_pdbs/2MUM_dryrun.json",
        [],
    ),
    (
        "prepared_pdbs/2MUM_dryrun.pdb",
        "prepared_pdbs/2MUM_neutralized.json",
        [],
    ),
    (
        "prepared_pdbs/2MUM_neutralized.pdb",
        "prepared_pdbs/2MUM_dryrun.json",
        [],
    ),
    (
        "prepared_pdbs/2MUM_blowup.pdb",
        "prepared_pdbs/2MUM_neutralized.json",
        [],
    ),
    (
        "prepared_pdbs/3h34_prepared.pdb",
        "prepared_pdbs/3h34_prepared.json",
        [],
    ),
    (
        "1A4T.pdb",
        "1A4T.json",
        [],
    ),
    (
        "1A4T.cif",
        "1A4T.json",
        [],
    ),
    (
        "prepared_pdbs/1a4t_samechain.pdb",
        "1A4T.json",
        [],
    ),
]
SLOW_PDBS: list[tuple[str, str, list[ResidueDefinition]]] = [
    (
        "prepared_pdbs/5eil_fixed.pdb",
        "prepared_pdbs/5eil_fixed.json",
        [],
    ),
]


@pytest.mark.parametrize(
    "fn_stem",
    [
        "2MUM_letters_in_resseq",
        "2MUM_letters_in_serial",
        "2MUM_reuse_resseq",  # Fails because adjacent PROs can't be distinguished
        "2MUM_reuse_serial",
        "2MUM_icode",
        "2MUM_discontiguous_serial",
        "2MUM_discontiguous_resseq",
        "2MUM_composed_function",
    ],
)
def test_extended_atom_residue_numbering(
    fn_stem: str,
    tmp_ccd_cache: CcdCache,
):
    path_stem = Path("prepared_pdbs") / fn_stem
    tmp_ccd_cache.auto_download = True
    topology_identical_to_jsontop(
        path_stem.with_suffix(".pdb"),
        path_stem.with_suffix(".json"),
        [],
        tmp_ccd_cache,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("pdbfile", "jsontopfile", "additional_definitions"),
    SLOW_PDBS,
)
def test_topology_identical_to_jsontop_slow(
    pdbfile: str,
    jsontopfile: str,
    additional_definitions: list[ResidueDefinition],
    tmp_ccd_cache: CcdCache,
):
    tmp_ccd_cache.auto_download = True
    topology_identical_to_jsontop(
        pdbfile,
        jsontopfile,
        additional_definitions,
        tmp_ccd_cache,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "pdbfile",
    [
        file
        for dir in get_test_data_path("polymers/").iterdir()
        if dir.is_dir()
        for file in dir.iterdir()
        if (
            file.suffix.endswith(".pdb")
            and file.with_suffix(".topology.json").exists()
            and file.with_suffix(".monomers.json").exists()
        )
    ],
    ids=lambda pdbfile: pdbfile.name,
)
def test_polymers(
    pdbfile: Path,
    tmp_ccd_cache: CcdCache,
):
    jsontopfile: Path = pdbfile.with_suffix(".topology.json")
    monomersfile: Path = pdbfile.with_suffix(".monomers.json")
    all_smarts = [
        (key, smarts)
        for key, smarts_list in json.loads(monomersfile.read_text()).items()
        for smarts in smarts_list
    ]
    additional_definitions = [
        ResidueDefinition._anon_from_smarts(smarts, description=key)
        for key, smarts in all_smarts
    ]
    topology_identical_to_jsontop(
        pdbfile,
        jsontopfile,
        additional_definitions,
        tmp_ccd_cache,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "pdbfile",
    [
        file
        for dir in get_test_data_path("polymers/").iterdir()
        if dir.is_dir()
        for file in dir.iterdir()
        if (
            file.suffix.endswith(".pdb")
            and file.with_suffix(".topology.json").exists()
            and file.with_suffix(".monomers.smiles.json").exists()
        )
    ],
    ids=lambda pdbfile: pdbfile.name,
)
def test_polymers_smiles(
    pdbfile: Path,
    tmp_ccd_cache: CcdCache,
):
    jsontopfile: Path = pdbfile.with_suffix(".topology.json")
    monomersfile: Path = pdbfile.with_suffix(".monomers.smiles.json")
    all_smiles = [
        (key, smiles)
        for key, smiles_list in json.loads(monomersfile.read_text()).items()
        for smiles in smiles_list
    ]
    additional_definitions = [
        ResidueDefinition.anon_from_smiles_marked_nonleaving(smiles, description=key)
        for key, smiles in all_smiles
    ]
    topology_identical_to_jsontop(
        pdbfile,
        jsontopfile,
        additional_definitions,
        tmp_ccd_cache,
    )


@pytest.mark.parametrize(
    ("pdbfile", "jsontopfile", "additional_definitions"),
    FAST_PDBS,
)
def test_topology_identical_to_jsontop_fast(
    pdbfile: str,
    jsontopfile: str,
    additional_definitions: list[ResidueDefinition],
    tmp_ccd_cache: CcdCache,
):
    tmp_ccd_cache.auto_download = True
    topology_identical_to_jsontop(
        pdbfile,
        jsontopfile,
        additional_definitions,
        tmp_ccd_cache,
    )


def topology_identical_to_jsontop(
    pdbfile: str | Path,
    jsontopfile: str | Path,
    additional_definitions: list[ResidueDefinition],
    residue_library: Mapping[str, Collection[ResidueDefinition]],
):
    pablo_top = topology_from_pdb(
        get_test_data_path(pdbfile),
        additional_definitions=additional_definitions,
        residue_library=residue_library,
    )
    jsontop_top = Topology.from_json(get_test_data_path(jsontopfile).read_text())

    assert pablo_top.n_molecules == jsontop_top.n_molecules
    for pablo_mol, jsontop_mol in zip(pablo_top.molecules, jsontop_top.molecules):
        assert pablo_mol.n_atoms == jsontop_mol.n_atoms
        for pablo_atom, jsontop_atom in zip(pablo_mol.atoms, jsontop_mol.atoms):
            assert pablo_atom.symbol == jsontop_atom.symbol
            assert pablo_atom.formal_charge == jsontop_atom.formal_charge
            assert pablo_atom.name == jsontop_atom.name
            # assert pablo_atom.metadata == jsontop_atom.metadata

        pablo_bonds = {
            (sort_tuple((bond.atom1_index, bond.atom2_index)), bond.bond_order)
            for bond in pablo_mol.bonds
        }
        jsontop_bonds = {
            (sort_tuple((bond.atom1_index, bond.atom2_index)), bond.bond_order)
            for bond in jsontop_mol.bonds
        }
        assert pablo_bonds == jsontop_bonds
