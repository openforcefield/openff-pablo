(chapter:additional_definitions)=
# `additional_definitions`: Residue-free chemical templates

The [`additional_definitions`] argument of the `topology_from_pdb` function is the easiest way to tell Pablo about new chemistries. It performs matches to PDB files by directly comparing chemistry, without considering residue or atom names. This allows it to match in several cases where the residue library fails:

1. Residue definitions that stretch across residue boundaries
2. When atom names are unknown, missing, or inconsistent
3. When the `crosslink` and `linking_bond` attributes don't adequately describe the connectivity of residues

The catch is that it *requires connectivity to already be defined*. Unlike the residue library, `additional_definitions` can only add chemical information to existing bonds; it cannot create bonds. It also requires atom elements to be correctly specified in the PDB file. To make this as flexible as possible, `additional definitions` can get bond information from two places:

1. From `CONECT` records in the PDB file, which specify which atoms are bonded but not the chemical properties of the bond, and
2. from residue definitions that have already matched from the residue library. Here, the chemical properties of the bond are specified, and so the additional definition must match those properties.

This is why `additional_definitions` is always checked *after* as many residues as possible are matched from the residue library; those residue matches may be essential for defining the connectivity of parts of an `additional_definition`.

Definitions in [`additional_definitions`] are often specified as anonymous definitions, as this generally simplifies definition creation and most of the other information is not used.

`additional_definitions` has one more catch: it can only *add information* to *unknown* atoms and bonds, and can never *disambiguate* matches from the residue library. This is a check against getting the wrong chemistry by accident!

[`additional_definitions`]: openff.pablo.topology_from_pdb.params.additional_definitions
