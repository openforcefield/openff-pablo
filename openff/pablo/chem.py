"""
Chemical information for re-use across residue databases.
"""

from openff.pablo.residue import BondDefinition

__all__ = [
    "DISULFIDE_BOND",
    "PEPTIDE_BOND",
    "PHOSPHODIESTER_BOND",
]

DISULFIDE_BOND = BondDefinition(
    atom1="SG",
    atom2="SG",
    order=1,
    aromatic=False,
    stereo=None,
)
"""
A single, non-aromatic bond between two atoms named ``SG``.

Disulfide bonds are single bonds between thiol groups commonly formed as
crosslinks in proteins between Cysteine residues.
"""

PEPTIDE_BOND = BondDefinition(
    atom1="C",
    atom2="N",
    order=1,
    aromatic=False,
    stereo=None,
)
"""
A single, non-aromatic bond between an atom named ``C`` and one named ``N``.

Peptide bonds are single bonds between amine and carboxylic acid groups that
constitute the polymer bonds between amino acid residues in proteins and
peptides.
"""

PHOSPHODIESTER_BOND = BondDefinition(
    atom1="O3'",
    atom2="P",
    order=1,
    aromatic=False,
    stereo=None,
)
"""
A single, non-aromatic bond between an atom named ``O3'`` and one named ``P``.

Phosphodiester bonds are single bonds between sugars and phosphates that
constitute the polymer bonds between nucleotide residues in nucleic acids.
"""
