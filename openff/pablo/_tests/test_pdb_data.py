from collections.abc import Iterable
from pathlib import Path

import pytest

from openff.pablo._pdb_data import PdbData, ResidueMatch
from openff.pablo._utils import default_dict
from openff.pablo.exceptions import UnknownOrAmbiguousSerialInConectError
from openff.pablo.residue import AtomDefinition, BondDefinition, ResidueDefinition


class TestResidueMatch:
    def test_get_atom_by_index(self, cys_match: ResidueMatch):
        for i, atom in enumerate(cys_match.residue_definition.atoms):
            assert cys_match.atom(i) == atom

    def test_get_atom_by_name(self, cys_match: ResidueMatch):
        for atom in cys_match.residue_definition.atoms:
            assert cys_match.atom(atom.name) == atom

    def test_get_atom_type_error(self, cys_match: ResidueMatch):
        with pytest.raises(TypeError):
            cys_match.atom(3.14159)  # type: ignore

    def test_set_crosslink_mutates_match(self, cys_match: ResidueMatch):
        cys_match.set_crosslink(cys_match.canonical_atom_name_to_index["SG"], 99)
        assert cys_match.crosslink == (cys_match.canonical_atom_name_to_index["SG"], 99)

    def test_set_crosslink_raises_on_wrong_atom(self, cys_match: ResidueMatch):
        with pytest.raises(ValueError):
            cys_match.set_crosslink(cys_match.canonical_atom_name_to_index["C"], 99)

    def test_prototype_index_in_match(self, cys_match: ResidueMatch):
        assert cys_match.prototype_index in cys_match.index_to_atomdef

    def test_set_crosslink_raises_on_atom1_from_other_residue(
        self,
        cys_match: ResidueMatch,
    ):
        with pytest.raises(ValueError):
            n_matched_atoms = len(cys_match.res_atom_idcs)
            cys_match.set_crosslink(n_matched_atoms + 1, n_matched_atoms + 2)

    def test_set_crosslink_raises_on_atom2_from_this_residue(
        self,
        cys_match: ResidueMatch,
    ):
        with pytest.raises(ValueError):
            cys_match.set_crosslink(cys_match.canonical_atom_name_to_index["SG"], 0)

    @pytest.mark.parametrize(
        "fixture_name, missing_atoms",
        [
            ("cys_match", []),
            ("cys_match_no_leaving", ["OXT", "HXT", "H2", "HG"]),
            ("cys_match_no_leaving_deprotonated_sidechain", ["OXT", "HXT", "H2"]),
        ],
    )
    def test_missing_atoms(
        self,
        request: pytest.FixtureRequest,
        fixture_name: str,
        missing_atoms: Iterable[str],
    ):
        match: ResidueMatch = request.getfixturevalue(fixture_name)
        assert match.missing_atoms == set(missing_atoms)

    def test_missing_leaving_atoms(
        self,
        hoh_def: ResidueDefinition,
    ):
        match = ResidueMatch(
            residue_definition=hoh_def,
            crosslink=None,
            index_to_atomdef={
                i: atom for i, atom in enumerate(hoh_def.atoms) if atom.name != "O"
            },
        )
        assert match.missing_atoms == {"O"}
        assert match.missing_leaving_atoms == set()

    @pytest.mark.parametrize(
        "expect_prop_suffix",
        ("prior_bond", "posterior_bond", "crosslink"),
    )
    def test_expect_props_false_when_not_supported_by_resdef(
        self,
        expect_prop_suffix: str,
        hoh_match: ResidueMatch,
    ):
        assert getattr(hoh_match, "expect_" + expect_prop_suffix) is False

    @pytest.mark.parametrize(
        "expect_prop_suffix",
        ("prior_bond", "posterior_bond", "crosslink"),
    )
    def test_expect_props_false_when_leaving_atoms_present(
        self,
        expect_prop_suffix: str,
        cys_match: ResidueMatch,
    ):
        assert getattr(cys_match, "expect_" + expect_prop_suffix) is False

    def test_expect_props_false_when_leaving_atoms_missing_from_resdef(
        self,
        cys_match_no_leaving_deprotonated_sidechain: ResidueMatch,
    ):
        assert cys_match_no_leaving_deprotonated_sidechain.expect_crosslink is False

    @pytest.mark.parametrize(
        "expect_prop_suffix",
        ("prior_bond", "posterior_bond", "crosslink"),
    )
    def test_expect_props_true_when_leaving_atoms_absent(
        self,
        expect_prop_suffix: str,
        cys_match_no_leaving: ResidueMatch,
    ):
        assert getattr(cys_match_no_leaving, "expect_" + expect_prop_suffix) is True

    def test_match_agrees_with_self(self, any_match: ResidueMatch):
        any_match.agrees_with(any_match)


class TestPdbData:
    def test_can_load_pdb_file_from_file_name(self, pdbfn: Path):
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

    def test_can_load_pdb_file_from_file_oobject(self, pdbfn: Path):
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

    def test_append_coord_line(self):
        data = PdbData()
        data._append_coord_line(
            "ATOM     16  HD2 LYS A   1      -1.806   9.969   9.991  1.00  0.00           H",
        )

        assert data == PdbData(
            model=[None],
            serial=[16],
            name=["HD2"],
            alt_loc=[""],
            res_name=["LYS"],
            chain_id=["A"],
            res_seq=[1],
            i_code=[" "],
            x=[-1.806],
            y=[9.969],
            z=[9.991],
            occupancy=[1.00],
            temp_factor=[0.00],
            element=["H"],
            charge=[None],
            terminated=[False],
            serial_to_index=default_dict(list, {16: [0]}),
            conects=[set()],
            cryst1_a=None,
            cryst1_b=None,
            cryst1_c=None,
            cryst1_alpha=None,
            cryst1_beta=None,
            cryst1_gamma=None,
        )

    def test_parse_waters_pdb(self):
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
            model=[1, 1, 1, 1, 1, 1],
            serial=[1, 2, 3, 4, 5, 6],
            name=["H1", "H2", "O", "H1", "H2", "O"],
            alt_loc=["", "", "", "", "", ""],
            res_name=["HOH", "HOH", "HOH", "HOH", "HOH", "HOH"],
            chain_id=["A", "A", "A", "A", "A", "A"],
            res_seq=[1, 1, 1, 2, 2, 2],
            i_code=[" ", " ", " ", " ", " ", " "],
            x=[-1.806, -2.806, -1.806, -1.806, -2.806, -1.806],
            y=[8.969, 9.969, 9.969, 8.969, 9.969, 9.969],
            z=[4.991, 4.991, 4.991, 9.991, 9.991, 9.991],
            occupancy=[1.00, 0.99, 1.00, 1.00, 1.00, 1.00],
            temp_factor=[0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            element=["H", "H", "O", "H", "H", "O"],
            charge=[None, None, None, None, None, None],
            terminated=[False, False, False, True, True, True],
            serial_to_index=default_dict(
                list,
                {1: [0], 2: [1], 3: [2], 4: [3], 5: [4], 6: [5]},
            ),
            conects=[{2}, {2}, {0, 1}, {5}, {5}, {3, 4}],
            cryst1_a=10.0,
            cryst1_b=10.0,
            cryst1_c=10.0,
            cryst1_alpha=90.0,
            cryst1_beta=90.0,
            cryst1_gamma=90.0,
        )

    def test_process_conects_produces_indices(self):
        serial_to_index: dict[int, list[int]] = {3: [0], 4: [1]}
        lines: list[str] = ["CONECT    3    4"]

        conects = PdbData._process_conects(
            lines,
            serial_to_index,
            conects=[set(), set()],
            model=[None, None],
        )

        assert conects == [{1}, {0}]

    def test_process_conects_works_with_multiple_models(self):
        serial_to_index: dict[int, list[int]] = {3: [0, 2], 4: [1, 3]}
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

    def test_process_conects_raises_when_ambiguous(self):
        serial_to_index: dict[int, list[int]] = {3: [0], 4: [1, 2]}
        lines: list[str] = ["CONECT    3    4"]

        with pytest.raises(UnknownOrAmbiguousSerialInConectError):
            PdbData._process_conects(
                lines,
                serial_to_index,
                conects=[set(), set(), set()],
                model=[None, None, None],
            )

    def test_process_conects_raises_when_serial_missing(self):
        serial_to_index: dict[int, list[int]] = {3: [0], 4: [1, 2]}
        lines: list[str] = ["CONECT    3    5"]

        with pytest.raises(UnknownOrAmbiguousSerialInConectError):
            PdbData._process_conects(
                lines,
                serial_to_index,
                conects=[set(), set(), set()],
                model=[None, None, None],
            )

    def test_residue_indices(self):
        data = PdbData(
            res_name=["HOH"] * 2 + ["GLY"] * 8,
            chain_id=["A"] * 4 + ["B"] * 6,
            res_seq=[1] * 6 + [2] * 4,
            i_code=[" "] * 8 + ["A"] * 2,
            model=[None] * 10,
            alt_loc=[""] * 10,
        )

        assert list(data.residue_indices) == [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (8, 9),
        ]

    def test_residue_indices_multimodel(self):
        data = PdbData(
            res_name=["HOH"] * 2 + ["GLY"] * 10,
            chain_id=["A"] * 4 + ["B"] * 8,
            res_seq=[1] * 6 + [2] * 6,
            i_code=[" "] * 8 + ["A"] * 4,
            model=[None] * 10 + [1] * 2,
            alt_loc=[""] * 12,
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
        self,
        hewl_data: PdbData,
        cys_def: ResidueDefinition,
    ):
        with pytest.raises(ValueError):
            hewl_data.subset_matches_residue([], cys_def)

    def test_subset_matches_residue_succeeds_when_all_atoms_present(
        self,
        cys_def: ResidueDefinition,
        cys_data: PdbData,
    ):
        assert (
            cys_data.subset_matches_residue(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                cys_def,
            )
            is not None
        )

    def test_subset_matches_residue_fails_on_multiply_matched_atom(self):
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
        assert (
            PdbData(
                name=["H1", "H"],
                charge=charges,
                element=elements,
            ).subset_matches_residue([0, 1], res_def)
            is None
        )
        assert (
            PdbData(
                name=["O", "H"],
                charge=charges,
                element=elements,
            ).subset_matches_residue([0, 1], res_def)
            is not None
        )

    def test_subset_matches_residue_succeeds_when_all_leaving_atoms_absent(
        self,
        cys_def: ResidueDefinition,
        cys_data: PdbData,
    ):
        leaving_atoms = {atom.name for atom in cys_def.atoms if atom.leaving}
        assert (
            cys_data.subset_matches_residue(
                [
                    i
                    for i, name in enumerate(cys_data.name)
                    if name not in leaving_atoms
                ],
                cys_def,
            )
            is not None
        )

    def test_subset_matches_residue_fails_when_nonleaving_atom_missing(
        self,
        cys_def: ResidueDefinition,
        cys_data: PdbData,
    ):
        # Missing atom 1 (CA), a non-leaving atom
        assert cys_data.name[1] == "CA"
        assert cys_def.name_to_atom["CA"].leaving is False
        assert (
            cys_data.subset_matches_residue(
                [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                cys_def,
            )
            is None
        )

    def test_subset_matches_residue_fails_when_linkage_leaving_atoms_partially_missing(
        self,
        cys_def: ResidueDefinition,
        cys_data: PdbData,
    ):
        # Missing atom 13 (HXT), one of two posterior bond leaving atoms
        assert cys_data.name[13] == "HXT"
        assert (
            cys_data.subset_matches_residue(
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                cys_def,
            )
            is None
        )

    @pytest.mark.parametrize("bond_name", ("prior_bond", "posterior_bond", "crosslink"))
    def test_subset_matches_residue_succeeds_when_posterior_bond_leaving_atoms_missing(
        self,
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
        assert cys_data.subset_matches_residue(subset, cys_def) is not None

    def test_subset_matches_residue_fails_on_element_mismatch(
        self,
        cys_def: ResidueDefinition,
        cys_data: PdbData,
    ):
        cys_data.element[5] = "Zr"
        assert (
            cys_data.subset_matches_residue(range(len(cys_def.atoms)), cys_def) is None
        )

    def test_subset_matches_residue_tolerates_none_charge_and_empty_element(self):
        resdef = ResidueDefinition(
            atoms=(AtomDefinition.with_defaults("H", "H", charge=1),),
            bonds=(),
            crosslink=None,
            linking_bond=None,
            description="",
            residue_name="HPL",
        )
        data = PdbData(name=["H"], element=[""], charge=[None])
        assert data.subset_matches_residue([0], resdef) is not None

    def test_subset_matches_residue_tolerates_wrong_case_element(self):
        resdef = ResidueDefinition(
            atoms=(AtomDefinition.with_defaults("H", "H", charge=1),),
            bonds=(),
            crosslink=None,
            linking_bond=None,
            description="",
            residue_name="HPL",
        )
        data = PdbData(name=["H"], element=["h"], charge=[1])
        assert data.subset_matches_residue([0], resdef) is not None

    def test_get_residue_matches_loads_vicinal_disulfide(
        self,
        vicinal_disulfide_data: PdbData,
        cys_def: ResidueDefinition,
        cys_def_deprotonated_sidechain: ResidueDefinition,
    ):
        match1, match2, *excess_matches = vicinal_disulfide_data.get_residue_matches(
            residue_database={"CYS": [cys_def, cys_def_deprotonated_sidechain]},
            additional_substructures=[],
        )
        assert len(excess_matches) == 0
        assert len(match1) == 1
        assert len(match2) == 1
        match1 = match1[0]
        match2 = match2[0]

        assert match1.crosslink is not None and match2.crosslink is not None
        sg1, sg2 = match1.crosslink
        assert match2.crosslink == (sg2, sg1)
        assert vicinal_disulfide_data.name[sg1] == "SG"
        assert vicinal_disulfide_data.name[sg2] == "SG"

        assert match1.missing_atoms == {"HXT", "HG", "OXT"}
        assert match2.missing_atoms == {"H2", "HG"}

        assert match1.expect_posterior_bond
        assert not match1.expect_prior_bond
        assert match1.expect_crosslink

        assert not match2.expect_posterior_bond
        assert match2.expect_prior_bond
        assert match2.expect_crosslink

        assert match1.residue_definition is cys_def
        assert match2.residue_definition is cys_def
