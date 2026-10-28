from collections.abc import Iterable

import pytest
from openff.toolkit import Molecule

from openff.pablo._matching import AdditionalDefMatch, ResidueMismatch
from openff.pablo._pdb_data import ResidueMatch
from openff.pablo._tests.utils import only_match
from openff.pablo._utils import unwrap
from openff.pablo.residue import AtomDefinition, ResidueDefinition


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

    def test_set_crosslink_mutates_match(self, cys_match_no_leaving: ResidueMatch):
        sg_idx = cys_match_no_leaving.canonical_atom_name_to_index["SG"]
        cys_match_no_leaving.set_crosslink(sg_idx, 99)
        assert cys_match_no_leaving.crosslink_idcs == (sg_idx, 99)

    def test_set_linkages_raise_without_linkage(self, cys_match: ResidueMatch):
        assert cys_match.linkages.leavers == frozenset()
        index = cys_match.canonical_atom_name_to_index
        with pytest.raises(ValueError, match="without a crosslink"):
            cys_match.set_crosslink(index["SG"], 99)
        with pytest.raises(ValueError, match="without a prior bond"):
            cys_match.set_prior_bond(99, index["N"])
        with pytest.raises(ValueError, match="without a posterior bond"):
            cys_match.set_posterior_bond(index["C"], 99)

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

    def test_missing_leaving_atoms(self, cys_match_no_leaving: ResidueMatch):
        assert cys_match_no_leaving.missing_leaving_atoms == {
            atom.name
            for atom in cys_match_no_leaving.residue_definition.atoms
            if atom.leaving
        }

    def test_reject_keeps_linkages(self, cys_match_no_leaving: ResidueMatch):
        mismatch = cys_match_no_leaving.reject("reason")
        assert mismatch.linkages == {cys_match_no_leaving.linkages}

    def test_create_split_rejects_every_linkage_when_none_fit(
        self,
        cys_def: ResidueDefinition,
    ):
        # Only half of the posterior bond's leaving fragment is missing
        mismatch = unwrap(
            ResidueMatch.create_split(
                cys_def,
                {i: atom for i, atom in enumerate(cys_def.atoms) if atom.name != "HXT"},
            ),
        )

        assert isinstance(mismatch, ResidueMismatch)
        assert mismatch.linkages == frozenset(cys_def._linkage_combinations)
        assert mismatch.reason == (
            "Missing leaving atoms (HXT) are not explained by any combination of"
            + " bonds to other residues"
        )

    def test_create_split_rejects_missing_non_leaving_atoms(
        self,
        hoh_def: ResidueDefinition,
    ):
        result = unwrap(
            ResidueMatch.create_split(
                hoh_def,
                {i: atom for i, atom in enumerate(hoh_def.atoms) if atom.name != "O"},
            ),
        )
        assert isinstance(result, ResidueMismatch)
        assert result.reason == "Expected atoms are missing: O"
        assert result.linkages == frozenset()

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

        match1 = only_match(
            residue_definition=resdef,
            index_to_atomdef={10: c, 11: h, 12: o, 13: ox},
        )
        match2 = only_match(
            residue_definition=resdef,
            index_to_atomdef={10: c, 11: h, 12: ox, 13: o},
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
        match1 = only_match(
            residue_definition=benzene,
            index_to_atomdef=dict(enumerate(benzene.atoms)),
        )
        match2 = only_match(
            residue_definition=benzene_alt_kekulization,
            index_to_atomdef=dict(enumerate(benzene_alt_kekulization.atoms)),
        )
        assert match1.agrees_with(match2), "alternate kekulization"

        # Incorrect elements
        match1 = only_match(
            residue_definition=benzene,
            index_to_atomdef=dict(enumerate(benzene.atoms)),
        )
        match2 = only_match(
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
        )
        assert not match1.agrees_with(match2), "incorrect elements"

        # incorrect connectivity (correct elements
        match1 = only_match(
            residue_definition=benzene,
            index_to_atomdef=dict(enumerate(benzene.atoms)),
        )
        match2 = only_match(
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
        )
        assert not match1.agrees_with(match2), "incorrect connectivity"

        # Incorrect total charge (correct elements and connectivity)
        match1 = only_match(
            residue_definition=benzene,
            index_to_atomdef=dict(enumerate(benzene.atoms)),
        )
        match2 = only_match(
            residue_definition=contrived_dicarbanion,
            index_to_atomdef=dict(enumerate(contrived_dicarbanion.atoms)),
        )
        assert not match1.agrees_with(match2), "incorrect total charge"


class TestSharedLinkingAtom:
    def test_create_split_one_match_per_linkage(
        self,
        shared_linking_atom_def: ResidueDefinition,
    ):
        index_to_atomdef = {
            i: atom
            for i, atom in enumerate(shared_linking_atom_def.atoms)
            if atom.name != "HC3"
        }

        results = ResidueMatch.create_split(shared_linking_atom_def, index_to_atomdef)
        matches = [result for result in results if isinstance(result, ResidueMatch)]
        mismatch = unwrap(
            result for result in results if isinstance(result, ResidueMismatch)
        )

        assert {
            (match.expects_posterior_bond, match.expects_crosslink) for match in matches
        } == {(True, False), (False, True)}
        assert mismatch.linkages == frozenset(
            shared_linking_atom_def._linkage_combinations,
        ) - {match.linkages for match in matches}
        assert mismatch.reason == (
            "Missing leaving atoms (HC3) are not explained by these combinations of"
            + " bonds to other residues"
        )

    def test_create_split_single_match_when_leaving_atoms_present(
        self,
        shared_linking_atom_def: ResidueDefinition,
    ):
        results = ResidueMatch.create_split(
            shared_linking_atom_def,
            dict(enumerate(shared_linking_atom_def.atoms)),
        )
        match = unwrap(result for result in results if isinstance(result, ResidueMatch))
        mismatch = unwrap(
            result for result in results if isinstance(result, ResidueMismatch)
        )

        assert match.linkages.leavers == frozenset()
        assert mismatch.reason == (
            "No leaving atoms are missing, but these combinations of bonds to other"
            + " residues remove some"
        )

    def additional_def_match(
        self,
        resdef: ResidueDefinition,
        partner_name: str,
        partner_idx: int,
    ) -> AdditionalDefMatch:
        """A match to ``resdef`` with C1 bonded to ``partner_idx`` and HC3 absent"""
        mapping = {i: atom for i, atom in enumerate(resdef.atoms) if atom.name != "HC3"}
        mapping[partner_idx] = AtomDefinition.with_defaults(
            partner_name,
            "",
            leaving=True,
        )
        linkages = unwrap(
            linkages
            for linkages in resdef._linkage_combinations
            if linkages.leavers == frozenset({"HC3"})
            and (linkages.crosslink == partner_name.endswith("_CROSSLINK"))
        )
        return AdditionalDefMatch.from_mapping(mapping, resdef, linkages)

    def test_additional_def_match_agrees_regardless_of_linkage(
        self,
        shared_linking_atom_def: ResidueDefinition,
    ):
        posterior = self.additional_def_match(
            shared_linking_atom_def,
            "N1_POSTERIOR",
            99,
        )
        crosslink = self.additional_def_match(
            shared_linking_atom_def,
            "X_CROSSLINK",
            99,
        )

        assert posterior.linkages != crosslink.linkages
        assert posterior.agrees_with(crosslink)
        assert crosslink.agrees_with(posterior)

    def test_additional_def_match_disagrees_on_external_partner(
        self,
        shared_linking_atom_def: ResidueDefinition,
    ):
        to_99 = self.additional_def_match(shared_linking_atom_def, "X_CROSSLINK", 99)
        to_98 = self.additional_def_match(shared_linking_atom_def, "X_CROSSLINK", 98)

        assert not to_99.agrees_with(to_98)

    def test_agrees_with_rejects_mixed_match_types(
        self,
        shared_linking_atom_def: ResidueDefinition,
    ):
        additional = self.additional_def_match(
            shared_linking_atom_def,
            "X_CROSSLINK",
            99,
        )
        residue = only_match(
            shared_linking_atom_def,
            dict(enumerate(shared_linking_atom_def.atoms)),
        )

        with pytest.raises(AssertionError):
            residue.agrees_with(additional)
        with pytest.raises(AssertionError):
            additional.agrees_with(residue)
