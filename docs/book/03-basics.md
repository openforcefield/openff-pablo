# A basic example

Phew, that was a lot of technical talk! If you breezed over it, here's the lowdown: Getting chemical information out of PDB files is hard. Pablo tries its best to make it easy.

Many PDB files using standard names to refer to standard residues "just work" in Pablo, as long as they aren't missing any atoms:

```py
topology_from_pdb("5ap1_prepared.pdb")
```

Even files that include ligands can load automatically, as long as the ligands are complete and named in the standard way.

```py
topology_from_pdb("3h34_prepared.pdb")
```

`topology_from_pdb` can also load from file-like objects. 1A4T is a rare example of a PDB file from the PDB that is chemically complete and can load without modification:

```py
from urllib.request import urlopen
with urlopen("https://files.rcsb.org/download/1A4T.pdb") as pdb_file_object:
    topology = topology_from_pdb(pdb_file_object)
```

## What are the standard names and residues?

<!--Describe the CCD.-->
