from io import StringIO
from sys import argv

from openmm.app import PDBFile
from pdbfixer import PDBFixer

from openff.pablo import topology_from_pdb

fname = argv[1]

fixer = PDBFixer(fname)

fixer.missingResidues = {}
fixer.findMissingAtoms()
fixer.addMissingAtoms()
fixer.addMissingHydrogens(pH=7.4)

with StringIO() as f:
    PDBFile.writeFile(fixer.topology, fixer.positions, f)

    f.seek(0)

    top = topology_from_pdb(f)

pos = top.get_positions()
pos -= pos.mean(axis=0)
top.set_positions(pos)
top.box_vectors = None

with StringIO() as f:
    top.to_file(f)
    s = f.getvalue()

print(s)
