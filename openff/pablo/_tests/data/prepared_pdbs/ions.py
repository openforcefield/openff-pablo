from openff.interchange.components._packmol import pack_box
from openff.toolkit import Molecule
from openff.units import unit

water = Molecule.from_smiles("O")
for i, atom in enumerate(water.atoms):
    atom.metadata["residue_name"] = "HOH"
    if atom.symbol == "O":
        atom.name = "O"
    else:
        atom.name = f"H{i}"

li = Molecule.from_smiles("[Li+]")
li.atom(0).name = "LI"
li.atom(0).metadata["residue_name"] = "LI"

na = Molecule.from_smiles("[Na+]")
na.atom(0).name = "NA"
na.atom(0).metadata["residue_name"] = "NA"

k = Molecule.from_smiles("[K+]")
k.atom(0).name = "K"
k.atom(0).metadata["residue_name"] = "K"

rb = Molecule.from_smiles("[Rb+]")
rb.atom(0).name = "RB"
rb.atom(0).metadata["residue_name"] = "RB"

cs = Molecule.from_smiles("[Cs+]")
cs.atom(0).name = "CS"
cs.atom(0).metadata["residue_name"] = "CS"

f = Molecule.from_smiles("[F-]")
f.atom(0).name = "F"
f.atom(0).metadata["residue_name"] = "F"

cl = Molecule.from_smiles("[Cl-]")
cl.atom(0).name = "CL"
cl.atom(0).metadata["residue_name"] = "CL"

br = Molecule.from_smiles("[Br-]")
br.atom(0).name = "BR"
br.atom(0).metadata["residue_name"] = "BR"

i = Molecule.from_smiles("[I-]")
i.atom(0).name = "I"
i.atom(0).metadata["residue_name"] = "I"

iod = Molecule.from_smiles("[I-]")
iod.atom(0).name = "I"
iod.atom(0).metadata["residue_name"] = "IOD"

molecules, n_copies = zip(
    (water, 555),
    (li, 1),
    (na, 1),
    (k, 1),
    (rb, 1),
    (cs, 1),
    (f, 1),
    (cl, 1),
    (br, 1),
    (i, 1),
    (iod, 1),
)

pack_box(
    molecules=molecules,
    number_of_copies=n_copies,
    target_density=1 * unit.g / unit.ml,
).to_file("ions.pdb")
