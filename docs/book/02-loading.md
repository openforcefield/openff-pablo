# How Pablo loads PDB files

Loading a PDB file with Pablo follows four steps:

1. Parse the file into an object that is convenient for the next steps
2. Assign chemical information from the [residue library] to the residues of the PDB file by matching their names and/or CONECT records
3. Assign chemical information from any [additional definitions] by looking for matches between them and the {term}`connectivity graph` implied by the previous steps and the `CONECT` records
4. Construct a [`Topology`] from the assigned chemical information if they fully and unambiguously describe the PDB file, or else raise an error

Chemical information is therefore built up by combining information in the PDB file with information in the residue library. To load a PDB file, at least one of the following conditions must be met for each atom record:

1. The atom is part of a residue that has the same residue name and atom names as a residue definition in the residue library (chemical info assigned in step 2)
2. The atom is part of a residue that has the same residue name, elements, and connectivity (described through `CONECT` records) as a residue definition in the residue library
3. The atom is covered by a residue definition in `additional_definitions`, and all bonds in that definition are specified either via CONECT records or by matches found in the residue library. In this case, the atom and residue names  of atoms covered only by additional definitions are not consulted.

In all cases, all sources of chemical information must agree, so that there is no ambiguity about what chemical is intended. Two sources of chemical information agree if they would generate molecules with identical connectivity, elements, and net formal charge. Note that this means that the resonance form produced is arbitrary and depends on the order of residue definitions in the library.

[`Topology`]: #openff.toolkit.topology.Topology
[residue library]: #openff.pablo.topology_from_pdb.params.residue_library
[additional definitions]: #openff.pablo.topology_from_pdb.params.additional_definitions

## PDBx/mmCIF files

Pablo has basic support for PDBx/mmCIF files. This works by reading a subset of data entries in the file that are equivalent to the columns of a PDB file's `ATOM/HETATM` records and passing that data through the same matching and topology construction pipeline used for PDB files. This ignores most of the chemical information in the PDBx/mmCIF file. If you have a use-case for the PDBx/mmCIF format that isn't covered by the existing functionality, please let us know --- if there's demand, we'll implement it!
