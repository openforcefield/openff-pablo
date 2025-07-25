from pathlib import Path

import pytest
from openff.toolkit import Molecule

from openff.pablo._matching import MoleculeMatch, NoResidueDefinitions, ResidueMismatch
from openff.pablo._pdb_data import PdbData, ResidueMatch
from openff.pablo._utils import default_dict
from openff.pablo.ccd import CCD_RESIDUE_DEFINITION_CACHE
from openff.pablo.exceptions import UnknownOrAmbiguousSerialInConectError
from openff.pablo.residue import AtomDefinition, BondDefinition, ResidueDefinition


def test_can_load_pdb_file_from_file_name(pdbfn: Path):
    data = PdbData.from_file(pdbfn)
    n_atom_records = len(
        [
            line
            for line in pdbfn.read_text().splitlines()
            if line.startswith("ATOM  ") or line.startswith("HETATM")
        ],
    )
    for field_name in [
        "name",
        "model",
        "serial",
        "alt_loc",
        "res_name",
        "chain_id",
        "res_seq",
        "i_code",
        "x",
        "y",
        "z",
        "occupancy",
        "temp_factor",
        "element",
        "charge",
        "terminated",
        "conects",
    ]:
        assert len(getattr(data, field_name)) == n_atom_records


def test_can_load_pdb_file_from_file_oobject(pdbfn: Path):
    with open(pdbfn) as f:
        data = PdbData.from_file_object(f)
    n_atom_records = len(
        [
            line
            for line in pdbfn.read_text().splitlines()
            if line.startswith("ATOM  ") or line.startswith("HETATM")
        ],
    )
    for field_name in [
        "name",
        "model",
        "serial",
        "alt_loc",
        "res_name",
        "chain_id",
        "res_seq",
        "i_code",
        "x",
        "y",
        "z",
        "occupancy",
        "temp_factor",
        "element",
        "charge",
        "terminated",
        "conects",
    ]:
        assert len(getattr(data, field_name)) == n_atom_records


def test_append_coord_line():
    data = PdbData()
    data._append_coord_line(
        "ATOM     16  HD2 LYS A   1      -1.806   9.969   9.991  1.00  0.00           H",
    )

    assert data == PdbData(
        line_no=[None],
        model=[None],
        serial=["16"],
        name=["HD2"],
        alt_loc=[""],
        res_name=["LYS"],
        chain_id=["A"],
        res_seq=["1"],
        i_code=[" "],
        x=[-1.806],
        y=[9.969],
        z=[9.991],
        occupancy=[1.00],
        temp_factor=[0.00],
        element=["H"],
        charge=[None],
        terminated=[False],
        serial_to_index=default_dict(list, {"16": [0]}),
        conects=[set()],
        cryst1_a=None,
        cryst1_b=None,
        cryst1_c=None,
        cryst1_alpha=None,
        cryst1_beta=None,
        cryst1_gamma=None,
    )


def test_parse_waters_pdb():
    pdblines = [
        "CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1 ",
        "MODEL        1",
        "ATOM      1  H1  HOH A   1      -1.806   8.969   4.991  1.00  0.00           H",
        "ATOM      2  H2  HOH A   1      -2.806   9.969   4.991  0.99  0.00           H",
        "ATOM      3  O   HOH A   1      -1.806   9.969   4.991  1.00  0.00           O",
        "HETATM    4  H1  HOH A   2      -1.806   8.969   9.991  1.00  0.00           H",
        "HETATM    5  H2  HOH A   2      -2.806   9.969   9.991  1.00  0.00           H",
        "HETATM    6  O   HOH A   2      -1.806   9.969   9.991  1.00  0.00           O",
        "TER       7      HOH A   2",
        "CONECT    1    3",
        "CONECT    2    3",
        "CONECT    3    1    2",
        "CONECT    4    6",
        "CONECT    5    6",
        "CONECT    6    4    5",
        "ENDMDL",
    ]
    data = PdbData.parse_pdb(pdblines)

    assert data == PdbData(
        line_no=[3, 4, 5, 6, 7, 8],
        model=[1, 1, 1, 1, 1, 1],
        serial=["1", "2", "3", "4", "5", "6"],
        name=["H1", "H2", "O", "H1", "H2", "O"],
        alt_loc=["", "", "", "", "", ""],
        res_name=["HOH", "HOH", "HOH", "HOH", "HOH", "HOH"],
        chain_id=["A", "A", "A", "A", "A", "A"],
        res_seq=["1", "1", "1", "2", "2", "2"],
        i_code=[" ", " ", " ", " ", " ", " "],
        x=[-1.806, -2.806, -1.806, -1.806, -2.806, -1.806],
        y=[8.969, 9.969, 9.969, 8.969, 9.969, 9.969],
        z=[4.991, 4.991, 4.991, 9.991, 9.991, 9.991],
        occupancy=[1.00, 0.99, 1.00, 1.00, 1.00, 1.00],
        temp_factor=[0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        element=["H", "H", "O", "H", "H", "O"],
        charge=[None, None, None, None, None, None],
        terminated=[False, False, False, False, False, True],
        serial_to_index=default_dict(
            list,
            {"1": [0], "2": [1], "3": [2], "4": [3], "5": [4], "6": [5]},
        ),
        conects=[{2}, {2}, {0, 1}, {5}, {5}, {3, 4}],
        cryst1_a=10.0,
        cryst1_b=10.0,
        cryst1_c=10.0,
        cryst1_alpha=90.0,
        cryst1_beta=90.0,
        cryst1_gamma=90.0,
    )


def test_process_conects_produces_indices():
    serial_to_index: dict[str, list[int]] = {"3": [0], "4": [1]}
    lines: list[str] = ["CONECT    3    4"]

    conects = PdbData._process_conects(
        lines,
        serial_to_index,
        conects=[set(), set()],
        model=[None, None],
    )

    assert conects == [{1}, {0}]


def test_process_conects_works_with_multiple_models():
    serial_to_index: dict[str, list[int]] = {"3": [0, 2], "4": [1, 3]}
    model = [1, 1, 2, 2]
    conects: list[set[int]] = [set(), set(), set(), set()]
    lines: list[str] = ["CONECT    3    4"]

    conects = PdbData._process_conects(
        lines,
        serial_to_index,
        conects=conects,
        model=model,
    )

    assert conects == [{1}, {0}, {3}, {2}]


def test_process_conects_raises_when_ambiguous():
    serial_to_index: dict[str, list[int]] = {"3": [0], "4": [1, 2]}
    lines: list[str] = ["CONECT    3    4"]

    with pytest.raises(UnknownOrAmbiguousSerialInConectError):
        PdbData._process_conects(
            lines,
            serial_to_index,
            conects=[set(), set(), set()],
            model=[None, None, None],
        )


def test_process_conects_raises_when_serial_missing():
    serial_to_index: dict[str, list[int]] = {"3": [0], "4": [1, 2]}
    lines: list[str] = ["CONECT    3    5"]

    with pytest.raises(UnknownOrAmbiguousSerialInConectError):
        PdbData._process_conects(
            lines,
            serial_to_index,
            conects=[set(), set(), set()],
            model=[None, None, None],
        )


def test_residue_indices():
    data = PdbData(
        res_name=["HOH"] * 2 + ["GLY"] * 10,
        chain_id=["A"] * 4 + ["B"] * 8,
        res_seq=["1"] * 6 + ["2"] * 6,
        i_code=[" "] * 8 + ["A"] * 4,
        model=[None] * 12,
        alt_loc=[""] * 12,
        terminated=[False] * 9 + [True] + [False] * 2,
    )

    assert list(data.residue_indices) == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (10, 11),
    ]


def test_residue_indices_multimodel():
    data = PdbData(
        res_name=["HOH"] * 2 + ["GLY"] * 10,
        chain_id=["A"] * 4 + ["B"] * 8,
        res_seq=["1"] * 6 + ["2"] * 6,
        i_code=[" "] * 8 + ["A"] * 4,
        model=[None] * 10 + [1] * 2,
        alt_loc=[""] * 12,
        terminated=[False] * 12,
    )

    with pytest.warns(match="Multi-model files not supported"):
        residues_indices = list(data.residue_indices)

    assert residues_indices == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
    ]


def test_subset_matches_residue_raises_on_empty(
    hewl_data: PdbData,
    cys_def: ResidueDefinition,
):
    with pytest.raises(ValueError):
        hewl_data.subset_matches_residue([], cys_def)


def test_subset_matches_residue_succeeds_when_all_atoms_present(
    cys_def: ResidueDefinition,
    cys_data: PdbData,
):
    assert cys_data.subset_matches_residue(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        cys_def,
    )


def test_subset_matches_residue_fails_on_multiply_matched_atom():
    res_def = ResidueDefinition(
        linking_bond=None,
        atoms=(
            AtomDefinition.with_defaults(name="H1", symbol="H", synonyms=["H"]),
            AtomDefinition.with_defaults(name="O", symbol="O", charge=-1),
        ),
        bonds=(BondDefinition.with_defaults(atom1="H1", atom2="O"),),
        crosslink=None,
        description="",
        residue_name="UNK",
    )
    charges: list[int | None] = [None, None]
    elements = ["", ""]
    assert not PdbData(
        name=["H1", "H"],
        charge=charges,
        element=elements,
    ).subset_matches_residue([0, 1], res_def)
    assert PdbData(
        name=["O", "H"],
        charge=charges,
        element=elements,
    ).subset_matches_residue([0, 1], res_def)


def test_subset_matches_residue_succeeds_when_all_leaving_atoms_absent(
    cys_def: ResidueDefinition,
    cys_data: PdbData,
):
    leaving_atoms = {atom.name for atom in cys_def.atoms if atom.leaving}
    assert cys_data.subset_matches_residue(
        [i for i, name in enumerate(cys_data.name) if name not in leaving_atoms],
        cys_def,
    )


def test_subset_matches_residue_fails_when_nonleaving_atom_missing(
    cys_def: ResidueDefinition,
    cys_data: PdbData,
):
    # Missing atom 1 (CA), a non-leaving atom
    assert cys_data.name[1] == "CA"
    assert cys_def.name_to_atom["CA"].leaving is False
    assert not cys_data.subset_matches_residue(
        [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        cys_def,
    )


def test_subset_matches_residue_fails_when_linkage_leaving_atoms_partially_missing(
    cys_def: ResidueDefinition,
    cys_data: PdbData,
):
    # Missing atom 13 (HXT), one of two posterior bond leaving atoms
    assert cys_data.name[13] == "HXT"
    assert not cys_data.subset_matches_residue(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        cys_def,
    )


@pytest.mark.parametrize("bond_name", ("prior_bond", "posterior_bond", "crosslink"))
def test_subset_matches_residue_succeeds_when_posterior_bond_leaving_atoms_missing(
    cys_def: ResidueDefinition,
    cys_data: PdbData,
    bond_name: str,
):
    subset = [
        i
        for i, name in enumerate(cys_data.name)
        if name not in getattr(cys_def, bond_name + "_leaving_atoms")
    ]
    assert len(subset) in [12, 13]
    assert cys_data.subset_matches_residue(subset, cys_def)


def test_subset_matches_residue_fails_on_element_mismatch(
    cys_def: ResidueDefinition,
    cys_data: PdbData,
):
    cys_data.element[5] = "Zr"
    assert not cys_data.subset_matches_residue(range(len(cys_def.atoms)), cys_def)


def test_subset_matches_residue_tolerates_none_charge_and_empty_element():
    resdef = ResidueDefinition(
        atoms=(AtomDefinition.with_defaults("H", "H", charge=1),),
        bonds=(),
        crosslink=None,
        linking_bond=None,
        description="",
        residue_name="HPL",
    )
    data = PdbData(name=["H"], element=[""], charge=[None])
    assert data.subset_matches_residue([0], resdef)


def test_subset_matches_residue_tolerates_wrong_case_element():
    resdef = ResidueDefinition(
        atoms=(AtomDefinition.with_defaults("H", "H", charge=1),),
        bonds=(),
        crosslink=None,
        linking_bond=None,
        description="",
        residue_name="HPL",
    )
    data = PdbData(name=["H"], element=["h"], charge=[1])
    assert data.subset_matches_residue([0], resdef)


def test_match_residues_loads_vicinal_disulfide(
    vicinal_disulfide_data: PdbData,
    cys_def: ResidueDefinition,
    cys_def_deprotonated_sidechain: ResidueDefinition,
):
    match1, match2, *excess_matches = (
        [p for p in residue_matches if isinstance(p, ResidueMatch)]
        for residue_matches in vicinal_disulfide_data.match_residues(
            residue_database={"CYS": [cys_def, cys_def_deprotonated_sidechain]},
            additional_substructures=[],
            unknown_molecules=[],
        )
    )
    assert len(excess_matches) == 0
    assert len(match1) == 1
    assert len(match2) == 1
    match1 = match1[0]
    match2 = match2[0]

    assert match1.crosslink_idcs is not None and match2.crosslink_idcs is not None
    sg1, sg2 = match1.crosslink_idcs
    assert match2.crosslink_idcs == (sg2, sg1)
    assert vicinal_disulfide_data.name[sg1] == "SG"
    assert vicinal_disulfide_data.name[sg2] == "SG"

    assert match1.missing_atoms == {"HXT", "HG", "OXT"}
    assert match2.missing_atoms == {"H2", "HG"}

    assert match1.expects_posterior_bond
    assert not match1.expects_prior_bond
    assert match1.expects_crosslink

    assert not match2.expects_posterior_bond
    assert match2.expects_prior_bond
    assert match2.expects_crosslink

    assert match1.residue_definition is cys_def
    assert match2.residue_definition is cys_def


def test_get_name_based_matches_one_per_residue(pdbfn: Path):
    data = PdbData.from_file(pdbfn)
    results = list(data.get_name_based_matches(CCD_RESIDUE_DEFINITION_CACHE))
    assert len(results) == len(list(data.residue_indices))
    assert data.res_idx is not None
    assert len(list(data.residue_indices)) == len(set(data.res_idx))
    assert len(data.res_idx) == len(data.serial)
    assert sorted(set(data.res_idx)) == list(range(data.res_idx[-1] + 1))
    assert len(results) == len(set(data.res_idx))
    assert len(results) == data.res_idx[-1] + 1


def test_get_name_based_matches_subsequence_contains_noresiduedefs_when_unmatched(
    he_data: PdbData,
):
    results = list(he_data.get_name_based_matches({}))
    assert len(results) == 1
    assert len(results[0]) == 1
    assert isinstance(results[0][0], NoResidueDefinitions)


def test_filter_on_polymer_linkages_yields_all_matches_simple(cys_data: PdbData):
    matches = list(cys_data.get_name_based_matches(CCD_RESIDUE_DEFINITION_CACHE))
    residue_matches = matches[0]
    residue_matches.append(
        MoleculeMatch(
            index_to_atomdef={i: None for i in residue_matches[0].res_atom_idcs},
            residue_definition=Molecule.from_smiles("N[C@@H](CS)C(=O)O"),
        ),
    )
    results = list(
        cys_data.filter_on_polymer_linkages(residue_matches, (), (), [residue_matches]),
    )

    for before, after in zip(residue_matches, results):
        if isinstance(before, ResidueMatch):
            assert isinstance(after, (ResidueMatch, ResidueMismatch))
        else:
            assert type(before) is type(after)
        assert before.residue_definition == after.residue_definition


def test_filter_on_polymer_linkages_yields_all_matches():
    from openff.pablo._tests.utils import get_test_data_path
    from openff.pablo.chem import PEPTIDE_BOND

    # Loading this PDB file with this augmented CCD cache will test all kinds
    # of residue-residue interface; see prepared_pdbs/polyglycines.py
    data = PdbData.from_file(get_test_data_path("prepared_pdbs/polyglycines.pdb"))
    matches = list(
        data.get_name_based_matches(
            CCD_RESIDUE_DEFINITION_CACHE.with_(
                [
                    ResidueDefinition.from_smiles(
                        mapped_smiles="[N-:1]([H:2])[C:3]([H:4])([H:5])[C:6](=[O:7])[O:8][H:9]",
                        atom_names={
                            1: "N",
                            2: "H",
                            3: "CA",
                            4: "HA1",
                            5: "HA2",
                            6: "C",
                            7: "O",
                            8: "OXT",
                            9: "HXT",
                        },
                        residue_name="GLY",
                        leaving_atoms=(8, 9),
                        linking_bond=PEPTIDE_BOND,
                        description="GLYCINE w/ negative formal charge on N",
                    ),
                    ResidueDefinition.from_smiles(
                        mapped_smiles="[N:1]([H:2])([H:8])[C:3]([H:4])([H:5])[C-:6]=[O:7]",
                        atom_names={
                            1: "N",
                            2: "H",
                            3: "CA",
                            4: "HA1",
                            5: "HA2",
                            6: "C",
                            7: "O",
                            8: "H2",
                        },
                        residue_name="GLY",
                        leaving_atoms=(8,),
                        linking_bond=PEPTIDE_BOND,
                        description="GLYCINE w/ negative formal charge on C",
                    ),
                ],
            ),
        ),
    )

    for residue_matches in matches:
        results = list(
            data.filter_on_polymer_linkages(residue_matches, (), (), [residue_matches]),
        )

        for before, after in zip(residue_matches, results):
            if isinstance(before, ResidueMatch):
                assert isinstance(after, (ResidueMatch, ResidueMismatch))
            else:
                assert type(before) is type(after)
            assert before.residue_definition == after.residue_definition
