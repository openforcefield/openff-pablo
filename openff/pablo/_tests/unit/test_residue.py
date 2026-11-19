import pytest
from openff.toolkit import Molecule

from openff.pablo._tests.utils import get_test_data_path
from openff.pablo._utils import unwrap
from openff.pablo.ccd._ccdcache import CcdCache
from openff.pablo.chem import DISULFIDE_BOND, PEPTIDE_BOND
from openff.pablo.exceptions import ResidueValidationError
from openff.pablo.residue import (
    AtomDefinition,
    BondDefinition,
    ResidueDefinition,
    _skip_residue_definition_validation,
)


class TestBondDefinition:
    def test_flipped(self):
        assert (
            BondDefinition.with_defaults("H1", "H2")
            == BondDefinition.with_defaults("H2", "H1").flipped()
        )

    def test_replace_atom1(self):
        with _skip_residue_definition_validation():
            replaced = PEPTIDE_BOND.replace(atom1="CO")

        assert PEPTIDE_BOND.atom1 == "C"
        assert replaced.atom1 == "CO"
        assert PEPTIDE_BOND.atom2 == replaced.atom2
        assert PEPTIDE_BOND.order == replaced.order
        assert PEPTIDE_BOND.aromatic == replaced.aromatic
        assert PEPTIDE_BOND.stereo == replaced.stereo

    def test_replace_atom2(self):
        with _skip_residue_definition_validation():
            replaced = PEPTIDE_BOND.replace(atom2="NH")

        assert PEPTIDE_BOND.atom2 == "N"
        assert replaced.atom2 == "NH"
        assert PEPTIDE_BOND.atom1 == replaced.atom1
        assert PEPTIDE_BOND.order == replaced.order
        assert PEPTIDE_BOND.aromatic == replaced.aromatic
        assert PEPTIDE_BOND.stereo == replaced.stereo


class TestResidueDefinition:
    @pytest.fixture
    def cys_mapped_smiles(self) -> str:
        return r"[N:1]([C@@:2]([C:5]([S:6][H:13])([H:11])[H:12])([C:3]([O:7][H:14])=[O:4])[H:10])([H:8])[H:9]"

    @pytest.fixture
    def cys_mapped_atomnames(self) -> dict[int, str]:
        return {
            1: "N",
            2: "CA",
            3: "C",
            4: "O",
            5: "CB",
            6: "SG",
            7: "OXT",
            8: "H",
            9: "H2",
            10: "HA",
            11: "HB2",
            12: "HB3",
            13: "HG",
            14: "HXT",
        }

    def test_replace_residue_name(self, cys_def: ResidueDefinition):
        with _skip_residue_definition_validation():
            replaced = cys_def.replace(residue_name="CYX")

        assert cys_def.residue_name == "CYS"
        assert replaced.residue_name == "CYX"
        assert cys_def.atoms == replaced.atoms
        assert cys_def.bonds == replaced.bonds
        assert cys_def.description == replaced.description
        assert cys_def.crosslink == replaced.crosslink
        assert cys_def.linking_bond == replaced.linking_bond

    def test_replace_atoms(self, cys_def: ResidueDefinition):
        with _skip_residue_definition_validation():
            replaced = cys_def.replace(atoms=())

        assert cys_def.atoms != ()
        assert replaced.atoms == ()
        assert cys_def.bonds == replaced.bonds
        assert cys_def.description == replaced.description
        assert cys_def.crosslink == replaced.crosslink
        assert cys_def.linking_bond == replaced.linking_bond
        assert cys_def.residue_name == replaced.residue_name

    def test_replace_bonds(self, cys_def: ResidueDefinition):
        with _skip_residue_definition_validation():
            replaced = cys_def.replace(bonds=())

        assert cys_def.bonds != ()
        assert replaced.bonds == ()
        assert cys_def.atoms == replaced.atoms
        assert cys_def.description == replaced.description
        assert cys_def.crosslink == replaced.crosslink
        assert cys_def.linking_bond == replaced.linking_bond
        assert cys_def.residue_name == replaced.residue_name

    def test_replace_description(self, cys_def: ResidueDefinition):
        with _skip_residue_definition_validation():
            replaced = cys_def.replace(description="Hello there")

        assert replaced.description == "Hello there"
        assert cys_def.description == "CYSTEINE"
        assert cys_def.atoms == replaced.atoms
        assert cys_def.bonds == replaced.bonds
        assert cys_def.crosslink == replaced.crosslink
        assert cys_def.linking_bond == replaced.linking_bond
        assert cys_def.residue_name == replaced.residue_name

    def test_replace_crosslink(self, cys_def: ResidueDefinition):
        with _skip_residue_definition_validation():
            replaced = cys_def.replace(crosslink=PEPTIDE_BOND)

        assert replaced.crosslink == PEPTIDE_BOND
        assert cys_def.crosslink == DISULFIDE_BOND
        assert cys_def.atoms == replaced.atoms
        assert cys_def.bonds == replaced.bonds
        assert cys_def.description == replaced.description
        assert cys_def.linking_bond == replaced.linking_bond
        assert cys_def.residue_name == replaced.residue_name

    def test_replace_linking_bond(self, cys_def: ResidueDefinition):
        with _skip_residue_definition_validation():
            replaced = cys_def.replace(linking_bond=DISULFIDE_BOND)

        assert replaced.linking_bond == DISULFIDE_BOND
        assert cys_def.linking_bond == PEPTIDE_BOND
        assert cys_def.atoms == replaced.atoms
        assert cys_def.bonds == replaced.bonds
        assert cys_def.description == replaced.description
        assert cys_def.crosslink == replaced.crosslink
        assert cys_def.residue_name == replaced.residue_name

    def test_from_molecule(
        self,
        cys_def: ResidueDefinition,
        cys_mapped_smiles: str,
        cys_mapped_atomnames: dict[int, str],
    ):
        cysteine = Molecule.from_mapped_smiles(cys_mapped_smiles)
        for i, atom in enumerate(cysteine.atoms, start=1):
            atom.name = cys_mapped_atomnames[i]
            if i in {7, 14, 13, 9}:
                atom.metadata["leaving_atom"] = True

        from_molecule = ResidueDefinition.from_molecule(
            cysteine,
            "CYS",
            linking_bond=PEPTIDE_BOND,
            crosslink=DISULFIDE_BOND,
            description="CYSTEINE",
        )
        assert from_molecule == cys_def

    def test_from_capped_molecule(
        self,
        cys_def: ResidueDefinition,
        cys_mapped_smiles: str,
        cys_mapped_atomnames: dict[int, str],
    ):
        cysteine = Molecule.from_mapped_smiles(cys_mapped_smiles)
        for i, atom in enumerate(cysteine.atoms, start=1):
            atom.name = cys_mapped_atomnames[i]

        from_molecule = ResidueDefinition.from_capped_molecule(
            cysteine,
            residue_name="CYS",
            leaving_atom_indices={6, 13, 12, 8},
            linking_bond=PEPTIDE_BOND,
            crosslink=DISULFIDE_BOND,
            description="CYSTEINE",
        )
        assert from_molecule == cys_def

    def test_from_smiles(
        self,
        cys_def: ResidueDefinition,
        cys_mapped_smiles: str,
        cys_mapped_atomnames: dict[int, str],
    ):
        cys_from_smiles = ResidueDefinition.from_smiles(
            residue_name="CYS",
            mapped_smiles=cys_mapped_smiles,
            atom_names=cys_mapped_atomnames,
            leaving_atoms={7, 14, 13, 9},
            crosslink=DISULFIDE_BOND,
            linking_bond=PEPTIDE_BOND,
            description="CYSTEINE",
        )
        assert cys_from_smiles == cys_def

    def test_from_smiles_raises_on_bad_atom_names(
        self,
        cys_def: ResidueDefinition,
        cys_mapped_smiles: str,
        cys_mapped_atomnames: dict[int, str],
    ):
        # Missing atom name
        atom_names = {k: v for k, v in cys_mapped_atomnames.items() if k != 5}
        with pytest.raises(ValueError):
            ResidueDefinition.from_smiles(
                residue_name="CYS",
                mapped_smiles=cys_mapped_smiles,
                atom_names=atom_names,
                leaving_atoms={7, 14, 13, 9},
                crosslink=DISULFIDE_BOND,
                linking_bond=PEPTIDE_BOND,
                description="CYSTEINE",
            )

    def test_leaving_atoms_must_be_associated_with_bond(
        self,
        cys_mapped_smiles: str,
        cys_mapped_atomnames: dict[int, str],
    ):
        kwargs = dict(
            residue_name="CCL",
            mapped_smiles=cys_mapped_smiles,
            atom_names=cys_mapped_atomnames,
            crosslink=DISULFIDE_BOND,
            linking_bond=PEPTIDE_BOND,
            description="CYSTEINE",
        )
        ResidueDefinition.from_smiles(**kwargs)  # type:ignore[arg]
        ResidueDefinition.from_smiles(**kwargs, leaving_atoms={7, 14, 13, 9})  # type:ignore[arg]
        with pytest.raises(ValueError):
            ResidueDefinition.from_smiles(**kwargs, leaving_atoms=[14])  # type:ignore[arg]

    def test_clashing_names_forbidden(self):
        with pytest.raises(ValueError):
            ResidueDefinition(
                residue_name="HHG",
                atoms=(
                    AtomDefinition.with_defaults("H", "H"),
                    AtomDefinition.with_defaults("H", "H"),
                ),
                bonds=(),
                linking_bond=None,
                crosslink=None,
                description="",
                virtual_sites=(),
            )

    def test_to_openff_molecule(
        self,
        cys_def: ResidueDefinition,
        cys_mapped_smiles: str,
    ):
        cys_molecule = Molecule.from_mapped_smiles(cys_mapped_smiles)
        assert cys_def.to_openff_molecule() == cys_molecule

    @pytest.mark.parametrize(
        "resdef_fixture",
        [
            "hoh_def",  # no linking bond or crosslink
            "hoh_def_with_synonyms",
            "cys_def",  # linking bond and crosslink
        ],
    )
    def test_to_openff_molecule_roundtrip(
        self,
        resdef_fixture: str,
        request: pytest.FixtureRequest,
    ):
        resdef: ResidueDefinition = request.getfixturevalue(resdef_fixture)
        roundtripped = ResidueDefinition.from_molecule(resdef.to_openff_molecule())
        assert roundtripped == resdef

    def test_to_openff_molecule_roundtrip_ccd(self, tmp_ccd_cache: CcdCache):
        for resdef in tmp_ccd_cache["ALA"]:
            roundtripped = ResidueDefinition.from_molecule(resdef.to_openff_molecule())
            assert roundtripped == resdef

    def test_to_openff_molecule_anon_roundtrip(self, cys_def: ResidueDefinition):
        molecule = cys_def.to_openff_molecule()
        anon_def = ResidueDefinition.anon_from_molecule(molecule)

        assert anon_def.is_anonymous
        assert anon_def.linking_bond == PEPTIDE_BOND
        assert anon_def.crosslink == DISULFIDE_BOND
        assert anon_def.description == cys_def.description

        # Atoms in the linking bond and crosslink keep their names; all other
        # names are replaced by molecule indices
        assert [atom.name for atom in anon_def.atoms] == [
            atom.name if atom.name in {"N", "C", "SG"} else str(i)
            for i, atom in enumerate(cys_def.atoms)
        ]

        # Apart from names, the chemistry is unchanged
        index_to_name = {str(i): atom.name for i, atom in enumerate(cys_def.atoms)}
        renamed = anon_def.replace(
            atoms=[
                atom.replace(name=index_to_name.get(atom.name, atom.name))
                for atom in anon_def.atoms
            ],
            bonds=[
                bond.replace(
                    atom1=index_to_name.get(bond.atom1, bond.atom1),
                    atom2=index_to_name.get(bond.atom2, bond.atom2),
                )
                for bond in anon_def.bonds
            ],
        )
        assert renamed._is_isomorphic_to(cys_def)

        # Leaving atoms are assigned to the same bonds as in the original
        assert {
            index_to_name.get(name, name) for name in anon_def._prior_bond_leaving_atoms
        } == cys_def._prior_bond_leaving_atoms
        assert {
            index_to_name.get(name, name)
            for name in anon_def._posterior_bond_leaving_atoms
        } == cys_def._posterior_bond_leaving_atoms
        assert {
            index_to_name.get(name, name) for name in anon_def._crosslink_leaving_atoms
        } == cys_def._crosslink_leaving_atoms

    def test_anon_from_molecule_duplicate_linking_atom_name_raises(
        self,
        cys_def: ResidueDefinition,
    ):
        molecule = cys_def.to_openff_molecule()
        unwrap(atom for atom in molecule.atoms if atom.name == "CB").name = "C"

        with pytest.raises(
            ResidueValidationError,
            match="Multiple atoms with linking atom name 'C'",
        ):
            ResidueDefinition.anon_from_molecule(molecule)

    # cys_def has 14 atoms; SG (index 5) is the crosslink atom
    @pytest.mark.parametrize(
        ("renames", "linking_bond", "expected_names"),
        [
            pytest.param(
                {"N": "4"},
                BondDefinition.with_defaults("C", "4"),
                {0: "4", 2: "C", 4: "14", 5: "SG"},
                id="clashes_with_index",
            ),
            pytest.param(
                {},
                BondDefinition.with_defaults("C", "4"),
                {2: "C", 4: "14", 5: "SG"},
                id="absent_atom_clashes_with_index",
            ),
            pytest.param(
                {},
                BondDefinition.with_defaults("3", "4"),
                {3: "14", 4: "15", 5: "SG"},
                id="two_clashes",
            ),
            pytest.param(
                {},
                BondDefinition.with_defaults("4", "14"),
                {4: "15", 5: "SG"},
                id="replacement_skips_linking_atom_name",
            ),
        ],
    )
    def test_anon_from_molecule_index_names_skip_linking_atom_names(
        self,
        cys_def: ResidueDefinition,
        renames: dict[str, str],
        linking_bond: BondDefinition,
        expected_names: dict[int, str],
    ):
        molecule = cys_def.to_openff_molecule()
        for atom in molecule.atoms:
            atom.name = renames.get(atom.name, atom.name)
        molecule.properties["linking_bond"] = linking_bond

        anon_def = ResidueDefinition.anon_from_molecule(molecule)

        assert [atom.name for atom in anon_def.atoms] == [
            expected_names.get(i, str(i)) for i in range(len(cys_def.atoms))
        ]

    def test_anon_from_molecule_overwrites_duplicate_names(
        self,
        cys_def: ResidueDefinition,
    ):
        molecule = cys_def.to_openff_molecule()
        for atom in molecule.atoms:
            if atom.symbol == "H":
                atom.name = "H"

        anon_def = ResidueDefinition.anon_from_molecule(molecule)

        assert [atom.name for atom in anon_def.atoms] == [
            atom.name if atom.name in {"N", "C", "SG"} else str(i)
            for i, atom in enumerate(cys_def.atoms)
        ]

    @pytest.mark.parametrize("residue_name", ["ACE", "DA"])
    def test_anon_from_molecule_absent_linking_atom(
        self,
        tmp_ccd_cache: CcdCache,
        residue_name: str,
    ):
        """Caps and 5' nucleotides may lack one of their linking bond's atoms"""
        for resdef in tmp_ccd_cache[residue_name]:
            anon_def = ResidueDefinition.anon_from_molecule(
                resdef.to_openff_molecule(),
            )
            assert anon_def.linking_bond == resdef.linking_bond
            assert anon_def._can_form_prior_bond == resdef._can_form_prior_bond
            assert anon_def._can_form_posterior_bond == resdef._can_form_posterior_bond

    def test_name_to_atom(self, hoh_def_with_synonyms: ResidueDefinition):
        assert (
            hoh_def_with_synonyms.name_to_atom["O"]
            == hoh_def_with_synonyms.name_to_atom["O1"]
        )
        assert (
            hoh_def_with_synonyms.name_to_atom["H1"]
            == hoh_def_with_synonyms.name_to_atom["HA"]
        )
        assert (
            hoh_def_with_synonyms.name_to_atom["H2"]
            == hoh_def_with_synonyms.name_to_atom["HB"]
        )
        assert hoh_def_with_synonyms.name_to_atom["O1"].symbol == "O"
        assert hoh_def_with_synonyms.name_to_atom["HA"].symbol == "H"
        assert hoh_def_with_synonyms.name_to_atom["HB"].symbol == "H"
        assert hoh_def_with_synonyms.name_to_atom["O1"].name == "O"
        assert hoh_def_with_synonyms.name_to_atom["HA"].name == "H1"
        assert hoh_def_with_synonyms.name_to_atom["HB"].name == "H2"

    def test_atom_bonded_to(self, hoh_def: ResidueDefinition):
        assert set(hoh_def.atoms_bonded_to("O")) == {"H1", "H2"}
        assert set(hoh_def.atoms_bonded_to("H1")) == {"O"}
        assert set(hoh_def.atoms_bonded_to("H2")) == {"O"}

    @pytest.mark.parametrize(
        ("bond_name", "leaving_atoms"),
        [
            ("prior_bond", {"H2"}),
            ("posterior_bond", {"OXT", "HXT"}),
            ("crosslink", {"HG"}),
        ],
    )
    def test_leaving_atom_props(
        self,
        cys_def: ResidueDefinition,
        bond_name: str,
        leaving_atoms: set[str],
    ):
        assert getattr(cys_def, f"_{bond_name}_leaving_atoms") == leaving_atoms

    def test_prior_bond_linking_atom(
        self,
        cys_def: ResidueDefinition,
    ):
        assert cys_def._prior_bond_linking_atom == "N"

    def test_posterior_bond_linking_atom(
        self,
        cys_def: ResidueDefinition,
    ):
        assert cys_def._posterior_bond_linking_atom == "C"

    def test_is_isomorphic_to(
        self,
        cys_def: ResidueDefinition,
        cys_def_deprotonated_sidechain: ResidueDefinition,
    ):
        assert cys_def._is_isomorphic_to(cys_def)
        cys_def_shuffled_atoms = cys_def.replace(
            atoms=cys_def.atoms[5:] + cys_def.atoms[:5],
        )
        cys_def_shuffled_bonds = cys_def.replace(
            bonds=cys_def.bonds[5:] + cys_def.bonds[:5],
        )
        assert cys_def._is_isomorphic_to(cys_def_shuffled_atoms)
        assert cys_def != cys_def_shuffled_atoms
        assert cys_def._is_isomorphic_to(cys_def_shuffled_bonds)
        assert cys_def != cys_def_shuffled_bonds
        assert not cys_def._is_isomorphic_to(cys_def_deprotonated_sidechain)

    def test_deprotonated_at_cooh(self):
        cooh = ResidueDefinition.from_smiles(
            "[H:1][C:2](=[O:3])[O:4][H:5]",
            {1: "HC", 2: "C", 3: "O", 4: "OH", 5: "HO"},
            "COOH",
        )
        coo = ResidueDefinition.from_smiles(
            "[H:1][C:2](=[O:3])[O-:4]",
            {1: "HC", 2: "C", 3: "O", 4: "OH"},
            "COO",
        )

        assert cooh.deprotonated_at("HO")._is_isomorphic_to(coo)

    def test_protonated_at_cooh(self):
        cooh = ResidueDefinition.from_smiles(
            "[H:1][C:2](=[O:3])[O:4][H:5]",
            {1: "HC", 2: "C", 3: "O", 4: "OH", 5: "HO"},
            "COOH",
        )
        coo = ResidueDefinition.from_smiles(
            "[H:1][C:2](=[O:3])[O-:4]",
            {1: "HC", 2: "C", 3: "O", 4: "OH"},
            "COO",
        )

        assert coo.protonated_at("OH", "HO")._is_isomorphic_to(cooh)

    def test_protonated_at_oh(self):
        hydroxide_ion = ResidueDefinition(
            atoms=(
                AtomDefinition.with_defaults("O", "O", charge=-1),
                AtomDefinition.with_defaults("H1", "H"),
            ),
            bonds=(BondDefinition.with_defaults("H1", "O"),),
            crosslink=None,
            linking_bond=None,
            description="HYDROXIDE ION",
            residue_name="OH-",
            virtual_sites=(),
        )
        water_from_hydroxide = hydroxide_ion.protonated_at("O", "H2")
        assert water_from_hydroxide.to_openff_molecule().is_isomorphic_with(
            Molecule.from_smiles("O"),
        )

    def test_protonated_at_fails_on_synonym_clash(self):
        oh_resdef = ResidueDefinition(
            atoms=(
                AtomDefinition.with_defaults("O", "O", charge=-1),
                AtomDefinition.with_defaults("H1", "H", synonyms=["H2"]),
            ),
            bonds=(BondDefinition.with_defaults("H1", "O"),),
            crosslink=None,
            linking_bond=None,
            description="HYDROXIDE ION",
            residue_name="OH-",
            virtual_sites=(),
        )
        assert "H2" in oh_resdef.name_to_atom
        with pytest.raises(ValueError):
            oh_resdef.protonated_at("O", "H2")

    @pytest.mark.parametrize("ignore_synonym_clashes", [True, False])
    def test_protonated_at_fails_on_canonical_clash(
        self,
        ignore_synonym_clashes: bool,
    ):
        with pytest.raises(ValueError):
            ResidueDefinition(
                atoms=(
                    AtomDefinition.with_defaults("O", "O", charge=-1),
                    AtomDefinition.with_defaults("H", "H"),
                ),
                bonds=(BondDefinition.with_defaults("H", "O"),),
                crosslink=None,
                linking_bond=None,
                description="HYDROXIDE ION",
                residue_name="OH-",
                virtual_sites=(),
            ).protonated_at("O", "H", ignore_synonym_clashes=ignore_synonym_clashes)

    def test_deprotonated_at_cys_sidechain(
        self,
        cys_def: ResidueDefinition,
        cys_def_deprotonated_sidechain: ResidueDefinition,
    ):
        assert cys_def.deprotonated_at("HG")._is_isomorphic_to(
            cys_def_deprotonated_sidechain,
        )

    def test_deprotonated_at_gly_backbones(
        self,
        gly_def_neutral: ResidueDefinition,
        gly_def_zwitterionic: ResidueDefinition,
    ):
        assert gly_def_neutral.deprotonated_at("HXT")._is_isomorphic_to(
            gly_def_zwitterionic.deprotonated_at("H3"),
        )

    def test_protonated_at_gly_backbones(
        self,
        gly_def_neutral: ResidueDefinition,
        gly_def_zwitterionic: ResidueDefinition,
    ):
        assert gly_def_neutral.protonated_at("N", "H3")._is_isomorphic_to(
            gly_def_zwitterionic.protonated_at("OXT", "HXT"),
        )

    def test_vary_protonation_gly_backbones(
        self,
        gly_def_neutral: ResidueDefinition,
        gly_def_zwitterionic: ResidueDefinition,
    ):
        from_neutral = sorted(
            gly_def_neutral.vary_protonation(
                acidic=["HXT"],
                basic=[("N", "H3")],
            ),
            key=lambda resdef: sorted([atom.name for atom in resdef.atoms]),
        )
        from_zwitterionic = sorted(
            gly_def_zwitterionic.vary_protonation(
                acidic=["H3"],
                basic=[("OXT", "HXT")],
            ),
            key=lambda resdef: sorted([atom.name for atom in resdef.atoms]),
        )

        assert len(from_neutral) == 4
        assert len(from_zwitterionic) == 4

        for resdef_from_neutral, resdef_from_zwitterionic in zip(
            from_neutral,
            from_zwitterionic,
        ):
            assert resdef_from_zwitterionic._is_isomorphic_to(resdef_from_neutral)

    def test_virtual_sites_clash_with_names(self):
        with pytest.raises(ValueError, match="Virtual sites may not clash"):
            ResidueDefinition.from_smiles(
                "[H+:1]",
                atom_names={1: "H"},
                virtual_sites=["H"],
                residue_name="HPS",
            )

    def test_virtual_sites_clash_with_synonyms(self):
        resdef = ResidueDefinition.from_smiles(
            "[H+:1]",
            atom_names={1: "H"},
            virtual_sites=["EP"],
            residue_name="HPS",
        )
        with pytest.raises(ValueError, match="Virtual sites may not clash"):
            resdef.with_synonyms({"H": ["EP"]})

    def test_anon_from_sdf(self):
        sdf_path = get_test_data_path("3ip9_dye_trimmed.sdf")
        mol = Molecule.from_file(sdf_path, "SDF")
        resdef = ResidueDefinition.anon_from_sdf(sdf_path)
        assert resdef.to_openff_molecule() == mol

    def test_leaving_fragment_of_core_atom(self, cys_def: ResidueDefinition):
        assert sorted(cys_def._leaving_fragment_of("SG")) == ["HG"]
        assert sorted(cys_def._leaving_fragment_of("N")) == ["H2"]
        assert sorted(cys_def._leaving_fragment_of("C")) == ["HXT", "OXT"]

    def test_leaving_fragment_of_leaving_atom(self, cys_def: ResidueDefinition):
        assert sorted(cys_def._leaving_fragment_of("HG")) == ["HG"]
        assert sorted(cys_def._leaving_fragment_of("H2")) == ["H2"]
        assert sorted(cys_def._leaving_fragment_of("OXT")) == ["HXT", "OXT"]

    def test_validate_linking_atoms_have_single_leaving_fragment(
        self,
        tmp_ccd_cache: CcdCache,
    ):
        resdef = unwrap(
            resdef for resdef in tmp_ccd_cache["LYS"] if resdef.description == "LYSINE"
        )
        with pytest.raises(ResidueValidationError):
            resdef.replace(
                residue_name="LY6",
                crosslink=BondDefinition.with_defaults(atom1="NZ", atom2="CD"),
                atoms=[
                    (
                        atom.replace(leaving=True)
                        if atom.name in {"HZ2", "HZ3"}
                        else (
                            atom.replace(synonyms=(*atom.synonyms, "HZ"))
                            if atom.name == "HZ1"
                            else atom
                        )
                    )
                    for atom in resdef.atoms
                ],
                description=resdef.description + " w/ crosslink",
            )
