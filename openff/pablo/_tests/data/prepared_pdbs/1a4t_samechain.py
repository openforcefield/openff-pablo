from contextlib import chdir
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from pymol import cmd

PDBID = "1a4t"
OUTFILENAME = Path(f"{PDBID}_samechain.pdb")

with (
    TemporaryDirectory() as tmpdir,
    NamedTemporaryFile(suffix=".pdb", dir=tmpdir) as tmpfile,
    chdir(tmpdir),
):
    cmd.do(f"fetch {PDBID}, async=0")

cmd.do("remove not alt ''+A")
cmd.do("alter all, alt=''")
cmd.do("alter all, chain='A'")

cmd.set("pdb_use_ter_records", 0)
cmd.set("pdb_retain_ids", 1)
cmd.set("retain_order", 1)
cmd.save(OUTFILENAME)
