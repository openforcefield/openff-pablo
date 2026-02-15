import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from openff.toolkit import Topology

from openff.pablo._pdb import topology_from_pdb
from openff.pablo._tests.utils import get_test_data_path, topology_identical_to_jsontop
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
    (
        "1CSA.pdb",
        "1CSA.topology.json",
        [],
    ),
]
SLOW_PDBS: list[tuple[str, str, list[ResidueDefinition]]] = [
    (
        "prepared_pdbs/5eil_fixed.pdb",
        "prepared_pdbs/5eil_fixed.json",
        [],
    ),
    (
        "5ap1_prepared.pdb",
        "5ap1_prepared.json",
        [
            ResidueDefinition.anon_from_smiles(
                "O=C([O-])Cn1cc(cn1)c2ccc(cc2OCC#N)Nc3ccc(c(n3)NC4CCCCC4)C#N",
            ),
        ],
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

    pdbfile = path_stem.with_suffix(".pdb")
    jsontopfile = path_stem.with_suffix(".json")

    pablo_top = topology_from_pdb(
        get_test_data_path(pdbfile),
        residue_library=tmp_ccd_cache,
    )
    jsontop_top = Topology.from_json(get_test_data_path(jsontopfile).read_text())

    topology_identical_to_jsontop(pablo_top, jsontop_top)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("pdbfile", "jsontopfile", "additional_definitions"),
    [pytest.param(*args, marks=pytest.mark.slow) for args in SLOW_PDBS] + FAST_PDBS,
)
def test_topology_identical_to_jsontop_slow(
    pdbfile: str,
    jsontopfile: str,
    additional_definitions: list[ResidueDefinition],
    tmp_ccd_cache: CcdCache,
):
    tmp_ccd_cache.auto_download = True

    pablo_top = topology_from_pdb(
        get_test_data_path(pdbfile),
        additional_definitions=additional_definitions,
        residue_library=tmp_ccd_cache,
    )
    jsontop_top = Topology.from_json(get_test_data_path(jsontopfile).read_text())

    topology_identical_to_jsontop(
        pablo_top,
        jsontop_top,
    )


SLOW_POLYMERS = [
    "8d1b.pdb",
    "pnipam_modified-s49.pdb",
    "7wcc.pdb",
    "6cww.pdb",
    "1lyd.pdb",
    "PolyphenyleneIII.pdb",
    "8f0x.pdb",
    "polyethylene.pdb",
    "polyphenyleneI.pdb",
    "syntactic_styrene-s49.pdb",
    "peg_modified-s49.pdb",
    "polyethylmethacrylate-s81.pdb",
    "7qt2.pdb",
    "8e8i.pdb",
    "polyphenylenesulfone-s16.pdb",
    "7fse.pdb",
    "polyethylene-s9.pdb",
    "paam_modified-s64.pdb",
    "atactic_styrene-s9.pdb",
    "naturalrubber-s49.pdb",
    "polyphenyleneII.pdb",
    "8bhw.pdb",
    "8fy3.pdb",
    "7pvu.pdb",
    "polyvinylchloride-s81.pdb",
    "8gt9.pdb",
    "6mtg.pdb",
]

SKIP_POLYMERS = [
    "144d.pdb",  # All hydrogens come after all heavy atoms
    "133d.pdb",  # All hydrogens come after all heavy atoms
    "130d.pdb",  # All hydrogens come after all heavy atoms
    "122d.pdb",  # All hydrogens come after all heavy atoms
    "2q1r.pdb",  # All hydrogens come after all heavy atoms
    "7xjf.pdb",  # Non-standard arginine?
    "pnipam_modified-s49.pdb",  # Too slow
    "syntactic_styrene-s49.pdb",  # Too slow
    "8f0x.pdb",  # Too slow
    "peg_modified-s49.pdb",  # Too slow
    "polyethylmethacrylate-s81.pdb",  # Too slow
    "8fy3.pdb",  # Too slow
    "8bhw.pdb",  # Too slow
    "paam_modified-s64.pdb",  # Too slow
    "naturalrubber-s49.pdb",  # Too slow
]

XFAIL_POLYMERS = [
    "8ovp.pdb",  # BUG: Unmatched neighbour reads as unsupported bond
    "messy_sugar.pdb",  # Unknown additional_definitions matching issue
    "bisphenolA.pdb",  # Unknown additional_definitions matching issue
    "bip23267_sup-0002-appendixs1.pdb",  # Unknown additional_definitions matching issue
    "PAMAM.pdb",  # Unknown additional_definitions matching issue
    "polyvinylchloride.pdb",  # Unknown additional_definitions matching issue
    "polyvinylchloride-s81.pdb",  # Unknown additional_definitions matching issue
]


@pytest.mark.parametrize(
    "pdbfile",
    [
        pytest.param(
            file,
            marks=(
                *((pytest.mark.slow,) if file.name in SLOW_POLYMERS else ()),
                *((pytest.mark.skip,) if file.name in SKIP_POLYMERS else ()),
                *((pytest.mark.xfail,) if file.name in XFAIL_POLYMERS else ()),
            ),
        )
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
    recwarn: pytest.WarningsRecorder,
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
    return polymers(
        additional_definitions,
        pdbfile,
        jsontopfile,
        tmp_ccd_cache,
        recwarn,
    )


def polymers(
    additional_definitions: list[ResidueDefinition],
    pdbfile: Path,
    jsontopfile: Path,
    residue_library: Mapping[str, Sequence[ResidueDefinition]],
    recwarn: pytest.WarningsRecorder,
):
    jsontop_top = Topology.from_json(get_test_data_path(jsontopfile).read_text())
    pablo_top = topology_from_pdb(
        get_test_data_path(pdbfile),
        additional_definitions=additional_definitions,
        residue_library=residue_library,
    )

    # Skip if we raise an atom ordering warning
    # TODO: Regenerate topologies for PDB files that can't be represented as a Topology
    atom_order_warnings = [
        warning
        for warning in recwarn
        if (
            "Input PDB has an atom ordering that cannot be represented in an OpenFF Topology"
            in str(warning.message)
        )
    ]
    if len(atom_order_warnings) > 0:
        pytest.xfail("This JSONtop was mangled by the openff toolkit")

    return topology_identical_to_jsontop(pablo_top, jsontop_top)
