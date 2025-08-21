# How Pablo loads PDB files

PDB files are a highly permissive format widely used in molecular modelling. This permissiveness makes interpreting chemical information from PDB files complicated --- PDB files basically run on the principle of "you know what I mean". The OpenFF ecosystem relies on access to detailed chemical information of the modelled molecular system, and so Pablo is designed to be able to infer the chemistry of as many PDB files as possible while not introducing ambiguity, incorrect chemical inference, or dependence on structure to interpret chemistry.

Pablo therefore does not infer bonds from proximity, does not infer the presence of any missing atoms including hydrogens, and may fail to load PDB files that other software has no trouble with.

Pablo also requires a library of chemical information to match the PDB file against; the most common residues used in biomolecular modelling are shipped with Pablo, a much larger library provided by the RCSB PDB can be automatically fetched from via the internet, and users can also provide residue templates directly for unusual residues (or non-standard PDB files)

Loading a PDB file with Pablo follows three steps:

1. Parse the file into a data format convenient for the next steps
2. Match each residue against the residue definitions in the library for that residue using atom names, elements, and connectivity specified in the PDB file
3. Construct a `Topology` from the matched residue definitions if they fully and unambiguously describe the PDB file, or raise an error

### PDBx/mmCIF files

Pablo has basic support for PDBx/mmCIF files. This works by reading a subset of data entries in the file that are equivalent to the columns of a PDB file's `ATOM/HETATM` records and passing that data through the same matching and topology construction pipeline used for PDB files. This ignores most of the chemical information in the PDBx/mmCIF file. If you have a use-case for the PDBx/mmCIF format that isn't covered by the existing functionality, please let us know!
