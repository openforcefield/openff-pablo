"""
Patches to add essential features to the CCD.
"""

from openff.pablo.chem import DISULFIDE_BOND

from .._utils import unwrap
from ..residue import (
    AtomDefinition,
    BondDefinition,
    ResidueDefinition,
)
from ._ccdcache import PEPTIDE_BOND

__all__ = [
    "ACIDIC_PROTONS",
    "BASIC_ATOMS",
    "ATOM_NAME_SYNONYMS",
    "fix_caps",
    "add_protonation_variants",
    "add_synonyms",
    "disambiguate_alt_ids",
    "add_disulfide_crosslink",
    "add_dephosphorylated_5p_terminus",
    "patch_his_sidechain_zwitterion",
    "delete_doubly_deprotonated_arginine",
    "add_nh2_leaving_atom",
]


ACIDIC_PROTONS: dict[str, list[str]] = {
    "ALA": ["HXT", "H2"],
    "ARG": ["HXT", "H2", "HH12", "HH22"],
    "ASN": ["HXT", "H2"],
    "ASP": ["HXT", "H2", "HD2"],
    "CYS": ["HXT", "H2", "HG"],
    "GLN": ["HXT", "H2"],
    "GLU": ["HXT", "H2", "HE2"],
    "GLY": ["HXT", "H2"],
    # TODO: Special case HIS so that the neutral form is not a zwitterion
    "HIS": ["HXT", "H2", "HD1", "HE2"],
    "ILE": ["HXT", "H2"],
    "LEU": ["HXT", "H2"],
    "LYS": ["HXT", "H2", "HZ3"],
    "MET": ["HXT", "H2"],
    "PHE": ["HXT", "H2"],
    "PRO": ["HXT"],
    "SER": ["HXT", "H2", "HG"],
    "THR": ["HXT", "H2", "HG1"],
    "TRP": ["HXT", "H2", "HE1"],
    "TYR": ["HXT", "H2", "HH"],
    "VAL": ["HXT", "H2"],
    "DG": ["HOP2", "HOP3", "HO3'"],
    "DA": ["HOP2", "HOP3", "HO3'"],
    "DT": ["HOP2", "HOP3", "HO3'"],
    "DC": ["HOP2", "HOP3", "HO3'"],
    "G": ["HOP2", "HOP3", "HO3'"],
    "A": ["HOP2", "HOP3", "HO3'"],
    "U": ["HOP2", "HOP3", "HO3'"],
    "C": ["HOP2", "HOP3", "HO3'"],
}
"""Map from residue name to a list of atom names of abstractable hydrogens.

Each 3-tuple specifies an atom name to remove. This atom must have exactly one
bond. A variant residue definition is created with that atom and bond removed,
and the formal charge of the bonded atom reduced by one.

Note that all combinations of deprotonations are generated; this means a residue
with ``n`` abstractable hydrogens will have ``2**n`` variants."""


BASIC_ATOMS: dict[str, list[tuple[str, str]]] = {
    "ALA": [("N", "H3")],
    "ARG": [("N", "H3")],
    "ASN": [("N", "H3")],
    "ASP": [("N", "H3")],
    "CYS": [("N", "H3")],
    "GLN": [("N", "H3")],
    "GLU": [("N", "H3")],
    "GLY": [("N", "H3")],
    "HIS": [("N", "H3")],
    "ILE": [("N", "H3")],
    "LEU": [("N", "H3")],
    "LYS": [("N", "H3")],
    "MET": [("N", "H3")],
    "PHE": [("N", "H3")],
    "PRO": [],
    "SER": [("N", "H3")],
    "THR": [("N", "H3")],
    "TRP": [("N", "H3")],
    "TYR": [("N", "H3")],
    "VAL": [("N", "H3")],
}
"""Protonation variants that add an atom to the CCD.

Each 3-tuple specifies an atom name to protonate and the name of the added
proton. A variant residue definition is created with that atom and bond added,
and the formal charge of the atom increased by one."""

ATOM_NAME_SYNONYMS = {
    "NME": {
        "HN2": ["H"],
        "C": ["CH3"],
        "H1": ["1HH3"],
        "H2": ["2HH3"],
        "H3": ["3HH3"],
    },
    "ACE": {
        "H1": ["1HH3"],
        "H2": ["2HH3"],
        "H3": ["3HH3"],
    },
    "NA": {"NA": ["Na"]},
    "CL": {"CL": ["Cl"]},
    "ALA": {"H": ["H1"]},
    "ARG": {"H": ["H1"]},
    "ASN": {"H": ["H1"]},
    "ASP": {"H": ["H1"]},
    "CYS": {"H": ["H1"], "HB2": ["3HB"]},
    "GLN": {"H": ["H1"]},
    "GLU": {"H": ["H1"]},
    "GLY": {"H": ["H1"]},
    "HIS": {"H": ["H1"]},
    "ILE": {"H": ["H1"]},
    "LEU": {"H": ["H1"]},
    "LYS": {"H": ["H1"]},
    "MET": {"H": ["H1"]},
    "PHE": {"H": ["H1"]},
    "SER": {"H": ["H1"], "HB2": ["3HB"]},
    "THR": {"H": ["H1"]},
    "TRP": {"H": ["H1"]},
    "TYR": {"H": ["H1"]},
    "VAL": {"H": ["H1"]},
}
"""Map from residue name and then canonical atom name to a list of synonyms"""


def fix_caps(res: ResidueDefinition) -> list[ResidueDefinition]:
    """
    Fix ``"NON-POLYMER"`` residues so they can be used as caps for peptides.
    """

    return [
        res.replace(
            linking_bond=PEPTIDE_BOND,
            atoms=[
                atom.replace(
                    leaving=True if atom.name == "H" else atom.leaving,
                )
                for atom in res.atoms
            ],
        ),
    ]


def add_protonation_variants(res: ResidueDefinition) -> list[ResidueDefinition]:
    """
    Add protonation variants from the ACIDIC_PROTONS and BASIC_ATOMS constants

    Note that all combinations of protonations and deprotonations are generated;
    this means a residue with ``n`` abstractable hydrogens and ``m`` acidic atoms
    will have ``2**(n+m)`` variants.
    """
    return res.vary_protonation(
        acidic=ACIDIC_PROTONS.get(res.residue_name, []),
        basic=BASIC_ATOMS.get(res.residue_name, []),
        ignore_synonym_clashes=True,
    )


def add_synonyms(res: ResidueDefinition) -> list[ResidueDefinition]:
    """
    Patch a residue definition to include synonyms from :py:data:`ATOM_NAME_SYNONYMS`.
    """
    patched_residues = [
        res.replace(
            atoms=(
                atom.replace(
                    synonyms={
                        *atom.synonyms,
                        *ATOM_NAME_SYNONYMS.get(res.residue_name, {}).get(
                            atom.name,
                            [],
                        ),
                    },
                )
                for atom in res.atoms
            ),
        ),
    ]
    return patched_residues


def disambiguate_alt_ids(res: ResidueDefinition) -> list[ResidueDefinition]:
    """
    CCD patch: put alt atom ids in their own residue definitions if needed

    This patch should be run before other patches that add synonyms, as it
    assumes that there is at most one synonym (from the CCD alt id
    flag).

    Some CCD residues (like GLY) have alternative atom IDs that clash with
    canonical IDs for a different atom. This breaks synonyms because the
    clashing alternate ID is never assigned; the PDB file is interpreted as
    having two copies of the canonical ID atom. To fix this, we just split
    residue definitions with this clashing problem into two definitions, one
    with the canonical IDs and the other with the alternates.

    Note that if a linking atom is part of a clash, this may interfere with
    the disambiguated residue definition's ability to link to canonically named
    residue definitions.
    """
    clashes: list[int] = []
    canonical_names = {atom.name for atom in res.atoms}
    for i, atom in enumerate(res.atoms):
        for synonym in atom.synonyms:
            if synonym in canonical_names:
                clashes.append(i)

    if len(clashes) != 0:
        old_to_new: dict[str, str] = {}
        for atom in res.atoms:
            if atom.synonyms:
                old_to_new[atom.name] = unwrap(atom.synonyms)
            else:
                old_to_new[atom.name] = atom.name

        res1 = res.replace(
            atoms=[
                atom.replace(
                    synonyms=[],
                )
                for atom in res.atoms
            ],
        )
        res2 = res.replace(
            atoms=[
                atom.replace(
                    name=old_to_new[atom.name],
                    synonyms=[],
                )
                for atom in res.atoms
            ],
            bonds=[
                bond.replace(
                    atom1=old_to_new[bond.atom1],
                    atom2=old_to_new[bond.atom2],
                )
                for bond in res.bonds
            ],
            description=res.description + " altids",
            crosslink=(
                None
                if res.crosslink is None
                else res.crosslink.replace(
                    atom1=old_to_new[res.crosslink.atom1],
                    atom2=old_to_new.get(
                        res.crosslink.atom2,
                        res.crosslink.atom2,
                    ),
                )
            ),
            linking_bond=(
                None
                if res.linking_bond is None
                else res.linking_bond.replace(
                    atom1=old_to_new[res.linking_bond.atom1],
                    atom2=old_to_new.get(
                        res.linking_bond.atom2,
                        res.linking_bond.atom2,
                    ),
                )
            ),
        )
        return [res1, res2]
    else:
        return [res]


def add_disulfide_crosslink(res: ResidueDefinition) -> list[ResidueDefinition]:
    if not {"HG", "SG"}.issubset({atom.name for atom in res.atoms}):
        raise ValueError(
            "Can only add disulfide crosslink to residue with HG and SG atoms",
        )

    return [
        res.replace(
            crosslink=DISULFIDE_BOND,
            atoms=[
                atom if atom.name != "HG" else atom.replace(leaving=True)
                for atom in res.atoms
            ],
        ),
    ]


def add_dephosphorylated_5p_terminus(res: ResidueDefinition) -> list[ResidueDefinition]:
    phosphate_names = {"P", "HOP2", "HOP3", "OP1", "OP2", "OP3"}

    return [
        res,
        res.replace(
            atoms=(
                AtomDefinition(
                    name="HO5'",
                    synonyms=(),
                    symbol="H",
                    aromatic=False,
                    charge=0,
                    leaving=False,
                    stereo=None,
                ),
                *(atom for atom in res.atoms if atom.name not in phosphate_names),
            ),
            bonds=(
                BondDefinition(
                    atom1="O5'",
                    atom2="HO5'",
                    aromatic=False,
                    order=1,
                    stereo=None,
                ),
                *(
                    bond
                    for bond in res.bonds
                    if bond.atom1 not in phosphate_names
                    and bond.atom2 not in phosphate_names
                ),
            ),
            description=(
                res.description[:-17] + "-5'-PROTONATED"
                if res.description.endswith("-5'-MONOPHOSPHATE")
                else res.description + " 5' dephosphorylated"
            ),
        ),
    ]


def set_hop3_leaving(res: ResidueDefinition) -> list[ResidueDefinition]:
    """The OP3 and HOP3 atoms in nucleic acid residues are both absent when the
    residue forms part of a polymer, but only OP3 is marked as leaving; HOP3
    becomes a disconnected fragment"""
    # TODO: Replace this with disconnected fragment detection in ResidueDefinition._leaving_fragment_of()
    return [
        res.replace(
            atoms=(
                atom.replace(leaving=True) if atom.name == "HOP3" else atom
                for atom in res.atoms
            ),
        ),
    ]


def patch_his_sidechain_zwitterion(res: ResidueDefinition) -> list[ResidueDefinition]:
    nd1_atom = res.name_to_atom.get("ND1", None)
    ne2_atom = res.name_to_atom.get("NE2", None)

    if (
        nd1_atom is not None
        and nd1_atom.charge == 1
        and ne2_atom is not None
        and ne2_atom.charge == -1
    ):
        bonds: list[BondDefinition] = []
        nd1_ce1_bond: BondDefinition | None = None
        ne2_ce1_bond: BondDefinition | None = None
        for bond in res.bonds:
            if {bond.atom1, bond.atom2} == {"ND1", "CE1"}:
                nd1_ce1_bond = bond
            elif {bond.atom1, bond.atom2} == {"NE2", "CE1"}:
                ne2_ce1_bond = bond
            else:
                bonds.append(bond)

        if (
            nd1_ce1_bond is None
            or nd1_ce1_bond.order != 2
            or ne2_ce1_bond is None
            or ne2_ce1_bond.order != 1
        ):
            raise ValueError(
                "Zwitterionic histidine side chain detected but could not be corrected",
            )
        else:
            return [
                res.replace(
                    bonds=[
                        nd1_ce1_bond.replace(order=1),
                        ne2_ce1_bond.replace(order=2),
                        *bonds,
                    ],
                    atoms=[
                        nd1_atom.replace(charge=0),
                        ne2_atom.replace(charge=0),
                        *(
                            atom
                            for atom in res.atoms
                            if atom is not nd1_atom and atom is not ne2_atom
                        ),
                    ],
                ),
            ]
    else:
        return [res]


def delete_doubly_deprotonated_arginine(
    res: ResidueDefinition,
) -> list[ResidueDefinition]:
    """HH12 and HH22 are both acidic, but only one can leave at a time"""
    atom_names = {atom.name for atom in res.atoms}
    if "HH22" not in atom_names and "HH12" not in atom_names:
        return []
    else:
        return [res]


def add_nh2_leaving_atom(
    res: ResidueDefinition,
) -> list[ResidueDefinition]:
    return [
        res.replace(
            atoms=[
                *res.atoms,
                AtomDefinition.with_defaults(name="H3", symbol="H", leaving=True),
            ],
            bonds=[*res.bonds, BondDefinition.with_defaults(atom1="H3", atom2="N")],
            description=res.description + " +H3",
        ),
        res,
    ]
