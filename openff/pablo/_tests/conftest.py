from collections import defaultdict
from pathlib import Path

import pytest
from openff.toolkit import Molecule, Topology

from openff.pablo._pdb_data import PdbData, ResidueMatch
from openff.pablo._tests.utils import get_test_data_path
from openff.pablo.ccd import CcdCache
from openff.pablo.chem import DISULFIDE_BOND, PEPTIDE_BOND
from openff.pablo.residue import AtomDefinition, BondDefinition, ResidueDefinition


@pytest.fixture(
    params=[
        "5ap1_prepared.pdb",
        "prepared_pdbs/193l_prepared.pdb",
        "prepared_pdbs/2zuq_prepared.pdb",
        "prepared_pdbs/2hi7_prepared.pdb",
        "3cu9_vicinal_disulfide.pdb",
        "e2_7nel.pdb",
    ],
)
def pdbfn(request: pytest.FixtureRequest) -> Path:
    return get_test_data_path(request.param)


@pytest.fixture(
    params=[
        "1A4T.cif",
    ],
)
def pdbxfn(request: pytest.FixtureRequest) -> Path:
    return get_test_data_path(request.param)


@pytest.fixture
def tmp_ccd_cache(tmp_path: Path) -> CcdCache:
    import openff.pablo.ccd.patches as patches

    return CcdCache(
        # TODO: Use a proper resource setup for this
        library_paths=[Path(__file__).parent.parent / "ccd/data/ccd_cache"],
        cache_path=tmp_path,
        patches=[
            {
                "ACE": patches.fix_caps,
                "NME": patches.fix_caps,
                "NH2": patches.fix_caps,
                "CYS": patches.add_disulfide_crosslink,
            },
            {"*": patches.add_protonation_variants},
            {
                "U": patches.add_dephosphorylated_5p_terminus,
                "G": patches.add_dephosphorylated_5p_terminus,
                "C": patches.add_dephosphorylated_5p_terminus,
                "A": patches.add_dephosphorylated_5p_terminus,
                "DT": patches.add_dephosphorylated_5p_terminus,
                "DG": patches.add_dephosphorylated_5p_terminus,
                "DC": patches.add_dephosphorylated_5p_terminus,
                "DA": patches.add_dephosphorylated_5p_terminus,
                "NH2": patches.add_nh2_leaving_atom,
            },
            {
                "U": patches.set_hop3_leaving,
                "G": patches.set_hop3_leaving,
                "C": patches.set_hop3_leaving,
                "A": patches.set_hop3_leaving,
                "DT": patches.set_hop3_leaving,
                "DG": patches.set_hop3_leaving,
                "DC": patches.set_hop3_leaving,
                "DA": patches.set_hop3_leaving,
            },
            {"*": patches.disambiguate_alt_ids},
            {"*": patches.add_synonyms},
            {"*": patches.strip_linkless_leavers},
            {
                "HIS": patches.patch_his_sidechain_zwitterion,
                "ARG": patches.delete_doubly_deprotonated_arginine,
            },
        ],
        extra_definitions={
            "I": [ResidueDefinition.from_smiles("[I-:1]", {1: "I"}, "I")],
            "WAT": [
                ResidueDefinition.from_smiles(
                    "[H:2][O:1][H:3]",
                    {1: "O", 2: "H1", 3: "H2"},
                    "WAT",
                ).with_synonyms({"H1": ["1H"], "H2": ["2H"]}),
            ],
            # Maestro NME:
            "NMA": [
                ResidueDefinition.from_smiles(
                    residue_name="NMA",
                    mapped_smiles="[H:3][N:1]([H:4])[C:2]([H:5])([H:6])[H:7]",
                    atom_names={
                        1: "N",
                        2: "C",
                        3: "HN1",
                        4: "HN2",
                        5: "H1",
                        6: "H2",
                        7: "H3",
                    },
                    leaving_atoms=[3],
                    linking_bond=PEPTIDE_BOND,
                    description="METHYLAMINE (MAESTRO)",
                ).with_synonyms(
                    {
                        "HN2": ["H"],
                        "C": ["CA", "CH3"],
                        "H1": ["1HH3", "HA1", "1HA"],
                        "H2": ["2HH3", "HA2", "2HA"],
                        "H3": ["3HH3", "HA3", "3HA"],
                    },
                ),
            ],
        },
    )


@pytest.fixture
def all_aa_resnames() -> set[str]:
    resnames = {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLU",
        "GLN",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
    assert len(resnames) == 20
    return resnames


@pytest.fixture
def hewl_data() -> PdbData:
    return PdbData.from_file(
        get_test_data_path("prepared_pdbs/193l_prepared.pdb"),
    )


@pytest.fixture
def vicinal_disulfide_data() -> PdbData:
    return PdbData.from_file(
        get_test_data_path("3cu9_vicinal_disulfide.pdb"),
    )


@pytest.fixture
def vicinal_disulfide_molecule() -> Molecule:
    return Topology.from_pdb(
        get_test_data_path("3cu9_vicinal_disulfide.pdb"),
    ).molecule(0)


@pytest.fixture
def cys_pdblines() -> list[str]:
    return [
        "ATOM      1  N   CYS     2      -0.664  -1.578  -0.633  1.00  0.00           N  ",
        "ATOM      2  CA  CYS     2      -0.105  -0.229  -0.633  1.00  0.00           C  ",
        "ATOM      3  C   CYS     2       1.405  -0.269  -0.633  1.00  0.00           C  ",
        "ATOM      4  O   CYS     2       2.034  -1.337  -0.633  1.00  0.00           O  ",
        "ATOM      5  CB  CYS     2      -0.674   0.537   0.577  1.00  0.00           C  ",
        "ATOM      6  SG  CYS     2      -0.163   2.270   0.535  1.00  0.00           S  ",
        "ATOM      7  OXT CYS     2       2.131   0.963  -0.633  1.00  0.00           O  ",
        "ATOM      8  H   CYS     2      -0.036  -2.457  -0.633  1.00  0.00           H  ",
        "ATOM      9  H2  CYS     2      -1.244  -1.644   0.190  1.00  0.00           H  ",
        "ATOM     10  HA  CYS     2      -0.420   0.277  -1.564  1.00  0.00           H  ",
        "ATOM     11  HB2 CYS     2      -0.349   0.082   1.535  1.00  0.00           H  ",
        "ATOM     12  HB3 CYS     2      -1.778   0.509   0.585  1.00  0.00           H  ",
        "ATOM     13  HG  CYS     2       0.749   2.191   1.501  1.00  0.00           H  ",
        "ATOM     14  HXT CYS     2       2.294   1.239   0.272  1.00  0.00           H  ",
    ]


@pytest.fixture
def cys_data(cys_pdblines: list[str]) -> PdbData:
    return PdbData.parse_pdb(
        lines=cys_pdblines,
    )


@pytest.fixture
def e2_data() -> PdbData:
    return PdbData.from_file(get_test_data_path("e2_7nel.pdb"))


@pytest.fixture
def he_data() -> PdbData:
    return PdbData(
        line_no=[1],
        model=[None],
        serial=["1"],
        name=["HE"],
        alt_loc=[""],
        res_name=["HE"],
        chain_id=["A"],
        res_seq=["1"],
        i_code=[" "],
        x=[-1.806],
        y=[9.969],
        z=[9.991],
        occupancy=[1.00],
        temp_factor=[0.00],
        element=["He"],
        charge=[None],
        terminated=[False],
        serial_to_index=defaultdict(list, {"1": [0]}),
        conects=[set()],
        cryst1_a=None,
        cryst1_b=None,
        cryst1_c=None,
        cryst1_alpha=None,
        cryst1_beta=None,
        cryst1_gamma=None,
    )


@pytest.fixture
def cys_def() -> ResidueDefinition:
    atoms = (
        AtomDefinition.with_defaults(name="N", symbol="N"),
        AtomDefinition.with_defaults(name="CA", symbol="C", stereo="R"),
        AtomDefinition.with_defaults(name="C", symbol="C"),
        AtomDefinition.with_defaults(name="O", symbol="O"),
        AtomDefinition.with_defaults(name="CB", symbol="C"),
        AtomDefinition.with_defaults(name="SG", symbol="S"),
        AtomDefinition.with_defaults(name="OXT", symbol="O", leaving=True),
        AtomDefinition.with_defaults(name="H", symbol="H"),
        AtomDefinition.with_defaults(name="H2", symbol="H", leaving=True),
        AtomDefinition.with_defaults(name="HA", symbol="H"),
        AtomDefinition.with_defaults(name="HB2", symbol="H"),
        AtomDefinition.with_defaults(name="HB3", symbol="H"),
        AtomDefinition.with_defaults(name="HG", symbol="H", leaving=True),
        AtomDefinition.with_defaults(name="HXT", symbol="H", leaving=True),
    )

    bonds = (
        BondDefinition.with_defaults("N", "CA"),
        BondDefinition.with_defaults("N", "H"),
        BondDefinition.with_defaults("N", "H2"),
        BondDefinition.with_defaults("CA", "C"),
        BondDefinition.with_defaults("CA", "CB"),
        BondDefinition.with_defaults("CA", "HA"),
        BondDefinition.with_defaults("C", "O", order=2),
        BondDefinition.with_defaults("C", "OXT"),
        BondDefinition.with_defaults("CB", "SG"),
        BondDefinition.with_defaults("CB", "HB2"),
        BondDefinition.with_defaults("CB", "HB3"),
        BondDefinition.with_defaults("SG", "HG"),
        BondDefinition.with_defaults("OXT", "HXT"),
    )

    return ResidueDefinition(
        atoms=atoms,
        bonds=bonds,
        crosslink=DISULFIDE_BOND,
        linking_bond=PEPTIDE_BOND,
        description="CYSTEINE",
        residue_name="CYS",
        virtual_sites=(),
    )


@pytest.fixture
def cys_def_deprotonated_sidechain(
    cys_def: ResidueDefinition,
) -> ResidueDefinition:
    atoms: list[AtomDefinition] = []
    for atom in cys_def.atoms:
        if atom.name == "HG":
            pass
        elif atom.name == "SG":
            atoms.append(AtomDefinition.with_defaults(name="SG", symbol="S", charge=-1))
        else:
            atoms.append(atom)

    return ResidueDefinition(
        atoms=tuple(atoms),
        bonds=tuple(
            bond for bond in cys_def.bonds if "HG" not in [bond.atom1, bond.atom2]
        ),
        crosslink=DISULFIDE_BOND,
        linking_bond=PEPTIDE_BOND,
        description="CYSTEINE",
        residue_name="CYS",
        virtual_sites=(),
    )


@pytest.fixture
def hoh_def() -> ResidueDefinition:
    atoms = (
        AtomDefinition.with_defaults(name="H1", symbol="H"),
        AtomDefinition.with_defaults(name="H2", symbol="H"),
        AtomDefinition.with_defaults(name="O", symbol="O"),
    )

    bonds = (
        BondDefinition.with_defaults("O", "H2"),
        BondDefinition.with_defaults("O", "H1"),
    )

    return ResidueDefinition(
        atoms=atoms,
        bonds=bonds,
        crosslink=None,
        linking_bond=None,
        description="water",
        residue_name="HOH",
        virtual_sites=(),
    )


@pytest.fixture
def hooh_def() -> ResidueDefinition:
    return ResidueDefinition.from_smiles(
        "[H:1][O:2][O:3][H:4]",
        {1: "H1", 2: "O1", 3: "O2", 4: "H2"},
        "HOOH",
    )


@pytest.fixture
def hoh_def_with_synonyms() -> ResidueDefinition:
    atoms = (
        AtomDefinition.with_defaults(name="H1", symbol="H", synonyms=["HA"]),
        AtomDefinition.with_defaults(name="H2", symbol="H", synonyms=["HB"]),
        AtomDefinition.with_defaults(name="O", symbol="O", synonyms=["O1"]),
    )

    bonds = (
        BondDefinition.with_defaults("O", "H2"),
        BondDefinition.with_defaults("O", "H1"),
    )

    return ResidueDefinition(
        atoms=atoms,
        bonds=bonds,
        crosslink=None,
        linking_bond=None,
        description="water",
        residue_name="HOH",
        virtual_sites=(),
    )


@pytest.fixture
def gly_def_neutral() -> ResidueDefinition:
    atoms = (
        AtomDefinition.with_defaults(name="N", symbol="N"),
        AtomDefinition.with_defaults(name="CA", symbol="C", stereo="R"),
        AtomDefinition.with_defaults(name="C", symbol="C"),
        AtomDefinition.with_defaults(name="O", symbol="O"),
        AtomDefinition.with_defaults(name="HA1", symbol="H"),
        AtomDefinition.with_defaults(name="OXT", symbol="O", leaving=True),
        AtomDefinition.with_defaults(name="H", symbol="H"),
        AtomDefinition.with_defaults(name="H2", symbol="H", leaving=True),
        AtomDefinition.with_defaults(name="HA2", symbol="H"),
        AtomDefinition.with_defaults(name="HXT", symbol="H", leaving=True),
    )

    bonds = (
        BondDefinition.with_defaults("N", "CA"),
        BondDefinition.with_defaults("N", "H"),
        BondDefinition.with_defaults("N", "H2"),
        BondDefinition.with_defaults("CA", "C"),
        BondDefinition.with_defaults("CA", "HA1"),
        BondDefinition.with_defaults("CA", "HA2"),
        BondDefinition.with_defaults("C", "O", order=2),
        BondDefinition.with_defaults("C", "OXT"),
        BondDefinition.with_defaults("OXT", "HXT"),
    )

    return ResidueDefinition(
        atoms=atoms,
        bonds=bonds,
        crosslink=None,
        linking_bond=PEPTIDE_BOND,
        description="GLYCINE NEUTRAL",
        residue_name="GLY",
        virtual_sites=(),
    )


@pytest.fixture
def gly_def_zwitterionic() -> ResidueDefinition:
    atoms = (
        AtomDefinition.with_defaults(name="N", symbol="N", charge=1),
        AtomDefinition.with_defaults(name="CA", symbol="C", stereo="R"),
        AtomDefinition.with_defaults(name="C", symbol="C"),
        AtomDefinition.with_defaults(name="O", symbol="O"),
        AtomDefinition.with_defaults(name="HA1", symbol="H"),
        AtomDefinition.with_defaults(name="OXT", symbol="O", leaving=True, charge=-1),
        AtomDefinition.with_defaults(name="H", symbol="H"),
        AtomDefinition.with_defaults(name="H2", symbol="H", leaving=True),
        AtomDefinition.with_defaults(name="H3", symbol="H"),
        AtomDefinition.with_defaults(name="HA2", symbol="H"),
    )

    bonds = (
        BondDefinition.with_defaults("N", "CA"),
        BondDefinition.with_defaults("N", "H"),
        BondDefinition.with_defaults("N", "H2"),
        BondDefinition.with_defaults("N", "H3"),
        BondDefinition.with_defaults("CA", "C"),
        BondDefinition.with_defaults("CA", "HA1"),
        BondDefinition.with_defaults("CA", "HA2"),
        BondDefinition.with_defaults("C", "O", order=2),
        BondDefinition.with_defaults("C", "OXT"),
    )

    return ResidueDefinition(
        atoms=atoms,
        bonds=bonds,
        crosslink=None,
        linking_bond=PEPTIDE_BOND,
        description="GLYCINE ZWITTERION",
        residue_name="GLY",
        virtual_sites=(),
    )


@pytest.fixture
def cys_match(cys_def: ResidueDefinition) -> ResidueMatch:
    return ResidueMatch(
        residue_definition=cys_def,
        crosslink_idcs=None,
        index_to_atomdef={i: atom for i, atom in enumerate(cys_def.atoms)},
        vsite_idcs=(),
    )


@pytest.fixture
def cys_match_no_leaving(cys_def: ResidueDefinition) -> ResidueMatch:
    counter = iter(range(len(cys_def.atoms)))
    return ResidueMatch(
        residue_definition=cys_def,
        crosslink_idcs=None,
        index_to_atomdef={
            next(counter): atom for atom in cys_def.atoms if not atom.leaving
        },
        vsite_idcs=(),
    )


@pytest.fixture
def cys_match_no_leaving_deprotonated_sidechain(
    cys_def_deprotonated_sidechain: ResidueDefinition,
) -> ResidueMatch:
    counter = iter(range(len(cys_def_deprotonated_sidechain.atoms)))
    return ResidueMatch(
        residue_definition=cys_def_deprotonated_sidechain,
        crosslink_idcs=None,
        index_to_atomdef={
            next(counter): atom
            for atom in cys_def_deprotonated_sidechain.atoms
            if not atom.leaving
        },
        vsite_idcs=(),
    )


@pytest.fixture
def hoh_match(hoh_def: ResidueDefinition) -> ResidueMatch:
    return ResidueMatch(
        residue_definition=hoh_def,
        crosslink_idcs=None,
        index_to_atomdef={i: atom for i, atom in enumerate(hoh_def.atoms)},
        vsite_idcs=(),
    )


@pytest.fixture(
    params=[
        "hoh_match",
        "cys_match",
        "cys_match_no_leaving",
        "cys_match_no_leaving_deprotonated_sidechain",
    ],
)
def any_match(request: pytest.FixtureRequest) -> ResidueMatch:
    return request.getfixturevalue(request.param)
