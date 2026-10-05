from contextlib import chdir
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from pymol import cmd, stored

PDBID = "1hje"
OUTFILENAME = Path(f"{PDBID}_samechain.pdb")

with (
    TemporaryDirectory() as tmpdir,
    NamedTemporaryFile(suffix=".pdb", dir=tmpdir) as tmpfile,
    chdir(tmpdir),
):
    cmd.do(f"fetch {PDBID}, async=0")

cmd.do("remove not alt ''+A")
cmd.do("alter all, alt=''")

cmd.do("select resi 10 and resn LYS and name CD")
cmd.h_add("sele")
cmd.do("alter resi 10 and resn LYS and name H01, name='HD2'")
cmd.do("alter resi 10 and resn LYS and name H02, name='HD3'")
cmd.do("select resi 10 and resn LYS and name CE")
cmd.h_add("sele")
cmd.do("alter resi 10 and resn LYS and name H01, name='HE2'")
cmd.do("alter resi 10 and resn LYS and name H02, name='HE3'")
cmd.do("select resi 10 and resn LYS and name NZ")
cmd.h_add("sele")
cmd.do("alter resi 10 and resn LYS and name H01, name='HZ1'")
cmd.do("alter resi 10 and resn LYS and name H02, name='HZ2'")
cmd.do("alter resi 10 and resn LYS and name H03, name='HZ3'")

cmd.do("select resn HOH")
cmd.h_add("sele")
cmd.do("alter resn HOH and name H01, name='H1'")
cmd.do("alter resn HOH and name H02+H03, name='H2'")


cmd.set("pdb_use_ter_records", 0)
cmd.save(OUTFILENAME)

# Pymol doesn't write Disulfide CONECT records
assert OUTFILENAME.read_text().endswith("END\n")
OUTFILENAME.write_text(OUTFILENAME.read_text()[:-4])
with OUTFILENAME.open(mode="at") as f:
    stored.i_idcs = []
    cmd.do("select resn CYS and name SG and bound_to name SG")
    cmd.do("iterate sele, stored.i_idcs.append(index)")
    for i in stored.i_idcs:
        stored.j_idcs = []
        cmd.do(f"iterate name SG and bound_to index {i}, stored.j_idcs.append(index)")
        for j in stored.j_idcs:
            f.write(f"CONECT{i: >5}{j: >5}\n")
    f.write("END\n")
