from collections.abc import Iterable

import pytest
from openff.toolkit import Molecule

from openff.pablo._pdb_data import ResidueMatch
from openff.pablo.residue import ResidueDefinition


class TestResidueMatch:
    def test_get_atom_by_index(self, cys_match: ResidueMatch):
        for i, atom in enumerate(cys_match.residue_definition.atoms):
            assert cys_match.atom(i) == atom

    def test_get_atom_by_name(self, cys_match: ResidueMatch):
        for atom in cys_match.residue_definition.atoms:
            assert cys_match.atom(atom.name) == atom

    def test_get_atom_type_error(self, cys_match: ResidueMatch):
        with pytest.raises(TypeError):
            cys_match.atom(3.14159)  # pyright: ignore[reportArgumentType]

    def test_set_crosslink_mutates_match(self, cys_match: ResidueMatch):
        cys_match.set_crosslink(cys_match.canonical_atom_name_to_index["SG"], 99)
        assert cys_match.crosslink_idcs == (
            cys_match.canonical_atom_name_to_index["SG"],
            99,
        )

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
            crosslink_idcs=None,
            index_to_atomdef={
                i: atom for i, atom in enumerate(hoh_def.atoms) if atom.name != "O"
            },
            vsite_idcs=(),
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
        assert getattr(hoh_match, "expects_" + expect_prop_suffix) is False

    @pytest.mark.parametrize(
        "expect_prop_suffix",
        ("prior_bond", "posterior_bond", "crosslink"),
    )
    def test_expect_props_false_when_leaving_atoms_present(
        self,
        expect_prop_suffix: str,
        cys_match: ResidueMatch,
    ):
        assert getattr(cys_match, "expects_" + expect_prop_suffix) is False

    def test_expect_props_false_when_leaving_atoms_missing_from_resdef(
        self,
        cys_match_no_leaving_deprotonated_sidechain: ResidueMatch,
    ):
        assert cys_match_no_leaving_deprotonated_sidechain.expects_crosslink is False

    @pytest.mark.parametrize(
        "expect_prop_suffix",
        ("prior_bond", "posterior_bond", "crosslink"),
    )
    def test_expect_props_true_when_leaving_atoms_absent(
        self,
        expect_prop_suffix: str,
        cys_match_no_leaving: ResidueMatch,
    ):
        assert getattr(cys_match_no_leaving, "expects_" + expect_prop_suffix) is True

    def test_match_agrees_with_self(self, any_match: ResidueMatch):
        assert any_match.agrees_with(any_match)

    def test_match_agrees_with_carboxylate_resonance_form(self):
        mol = Molecule.from_smiles("C(=O)[O-]")
        mol.generate_unique_atom_names()
        resdef = ResidueDefinition.from_molecule(mol, residue_name="CAR")

        c, h, ox, o = sorted(
            resdef.atoms,
            key=lambda atom: (atom.symbol, atom.charge),
        )
        assert c.symbol == "C"
        assert h.symbol == "H"
        assert o.symbol == "O"
        assert o.charge == 0
        assert ox.symbol == "O"
        assert ox.charge == -1

        match1 = ResidueMatch(
            residue_definition=resdef,
            index_to_atomdef={10: c, 11: h, 12: o, 13: ox},
            vsite_idcs=(),
        )
        match2 = ResidueMatch(
            residue_definition=resdef,
            index_to_atomdef={10: c, 11: h, 12: ox, 13: o},
            vsite_idcs=(),
        )
        assert match1.agrees_with(match2)

    def test_match_agrees_with_benzene(self):
        mol = Molecule.from_mapped_smiles(
            "[C:1]1([H:2])=[C:3]([H:4])[C:5]([H:6])=[C:7]([H:8])[C:9]([H:10])=[C:11]([H:12])1",
        )
        mol.generate_unique_atom_names()
        benzene = ResidueDefinition.from_molecule(mol, residue_name="BEN")

        mol = Molecule.from_mapped_smiles(
            "[C:1]1([H:2])[C:3]([H:4])=[C:5]([H:6])[C:7]([H:8])=[C:9]([H:10])[C:11]([H:12])=1",
        )
        mol.generate_unique_atom_names()
        benzene_alt_kekulization = ResidueDefinition.from_molecule(
            mol,
            residue_name="BEN",
        )

        mol = Molecule.from_mapped_smiles(
            "[C-:1]1([H:2])[C-:3]([H:4])[C:5]([H:6])=[C:7]([H:8])[C:9]([H:10])=[C:11]([H:12])1",
        )
        mol.generate_unique_atom_names()
        contrived_dicarbanion = ResidueDefinition.from_molecule(
            mol,
            residue_name="BEN",
        )

        # Alternate kekulization
        match1 = ResidueMatch(
            residue_definition=benzene,
            index_to_atomdef=dict(enumerate(benzene.atoms)),
            vsite_idcs=(),
        )
        match2 = ResidueMatch(
            residue_definition=benzene_alt_kekulization,
            index_to_atomdef=dict(enumerate(benzene_alt_kekulization.atoms)),
            vsite_idcs=(),
        )
        assert match1.agrees_with(match2), "alternate kekulization"

        # Incorrect elements
        match1 = ResidueMatch(
            residue_definition=benzene,
            index_to_atomdef=dict(enumerate(benzene.atoms)),
            vsite_idcs=(),
        )
        match2 = ResidueMatch(
            residue_definition=benzene,
            index_to_atomdef={
                0: benzene.atoms[0],
                1: benzene.atoms[2],
                2: benzene.atoms[4],
                3: benzene.atoms[6],
                4: benzene.atoms[8],
                5: benzene.atoms[10],
                6: benzene.atoms[1],
                7: benzene.atoms[3],
                8: benzene.atoms[5],
                9: benzene.atoms[7],
                10: benzene.atoms[9],
                11: benzene.atoms[11],
            },
            vsite_idcs=(),
        )
        assert not match1.agrees_with(match2), "incorrect elements"

        # incorrect connectivity (correct elements
        match1 = ResidueMatch(
            residue_definition=benzene,
            index_to_atomdef=dict(enumerate(benzene.atoms)),
            vsite_idcs=(),
        )
        match2 = ResidueMatch(
            residue_definition=benzene,
            index_to_atomdef={
                0: benzene.atoms[0],
                1: benzene.atoms[1],
                2: benzene.atoms[4],
                3: benzene.atoms[5],
                4: benzene.atoms[6],
                5: benzene.atoms[7],
                6: benzene.atoms[2],
                7: benzene.atoms[3],
                8: benzene.atoms[8],
                9: benzene.atoms[9],
                10: benzene.atoms[10],
                11: benzene.atoms[11],
            },
            vsite_idcs=(),
        )
        assert not match1.agrees_with(match2), "incorrect connectivity"

        # Incorrect total charge (correct elements and connectivity)
        match1 = ResidueMatch(
            residue_definition=benzene,
            index_to_atomdef=dict(enumerate(benzene.atoms)),
            vsite_idcs=(),
        )
        match2 = ResidueMatch(
            residue_definition=contrived_dicarbanion,
            index_to_atomdef=dict(enumerate(contrived_dicarbanion.atoms)),
            vsite_idcs=(),
        )
        assert not match1.agrees_with(match2), "incorrect total charge"
