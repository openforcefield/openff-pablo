import inspect
from io import StringIO
from itertools import count
from pathlib import Path

from openff.toolkit import Topology


def main():
    in_fn = Path(__file__).parent / "2MUM_neutralized.json"
    out_fn_template = str(Path(__file__).parent / "2MUM_{}")

    # top = topology_from_pdb(in_fn)
    top = Topology.from_json(in_fn.read_text())

    dryrun(top, out_fn_template)

    reuse_serial(top, out_fn_template)
    reuse_resseq(top, out_fn_template)
    letters_in_serial(top, out_fn_template)
    letters_in_resseq(top, out_fn_template)
    # extended_width_serial(top, out_fn_template)
    # extended_width_resseq(top, out_fn_template)
    # icode(top, out_fn_template)
    discontiguous_serial(top, out_fn_template)
    discontiguous_resseq(top, out_fn_template)


def dryrun(top: Topology, out_fn_template: str):
    top = Topology(top)
    write_pdb_and_json(top, out_fn_template)


def write_pdb_and_json(top: Topology, out_fn_template: str):
    empty_serial_resseq_top = Topology(top)
    for atom in empty_serial_resseq_top.atoms:
        atom.metadata["atom_serial"] = 0
        atom.metadata["residue_number"] = "0"

    with StringIO() as f:
        empty_serial_resseq_top.to_file(f, file_format="PDB", keep_ids=True)
        pdbtext = f.getvalue()

    lines = []
    i = 0
    for line in pdbtext.splitlines():
        if line.startswith("ATOM  ") or line.startswith("HETATM"):
            line = (
                line[:6]
                + f"{top.atom(i).metadata['atom_serial']: >5}"
                + line[11:22]
                + f"{top.atom(i).metadata['residue_number']: >4}"
                + line[26:]
            )
            i += 1
        lines.append(line)

    out_fn_stem = Path(out_fn_template.format(inspect.stack()[1][3]))
    out_fn_stem.with_suffix(".pdb").write_text("\n".join(lines))
    out_fn_stem.with_suffix(".json").write_text(top.to_json())


def reuse_serial(top: Topology, out_fn_template: str):
    top = Topology(top)

    res_iter = iter(top.residues)
    for i in range(5):
        reused_serial = next(res_iter).atom(-1).metadata["atom_serial"]
    for res in res_iter:
        for atom in res.atoms:
            atom.metadata["atom_serial"] = reused_serial

    write_pdb_and_json(top, out_fn_template)


def reuse_resseq(top: Topology, out_fn_template: str):
    top = Topology(top)

    res_iter = iter(top.residues)
    for i in range(5):
        reused_resseq = next(res_iter).identifier[1]
    for res in res_iter:
        for atom in res.atoms:
            atom.metadata["residue_number"] = reused_resseq

    write_pdb_and_json(top, out_fn_template)


def letters_in_serial(top: Topology, out_fn_template: str):
    top = Topology(top)

    for atom in top.atoms:
        serial = str(atom.metadata["atom_serial"])
        assert len(serial) <= 4
        atom.metadata["atom_serial"] = f"A{serial:0>4}"

    write_pdb_and_json(top, out_fn_template)


def letters_in_resseq(top: Topology, out_fn_template: str):
    top = Topology(top)

    for atom in top.atoms:
        resseq = atom.metadata["residue_number"]
        assert len(resseq) <= 3
        atom.metadata["residue_number"] = f"A{resseq:0>3}"

    write_pdb_and_json(top, out_fn_template)


def extended_width_serial(top: Topology, out_fn_template: str):
    top = Topology(top)

    write_pdb_and_json(top, out_fn_template)


def extended_width_resseq(top: Topology, out_fn_template: str):
    top = Topology(top)

    write_pdb_and_json(top, out_fn_template)


def icode(top: Topology, out_fn_template: str):
    top = Topology(top)

    write_pdb_and_json(top, out_fn_template)


def discontiguous_serial(top: Topology, out_fn_template: str):
    top = Topology(top)

    def increment_every_n(n: int):
        for i in count():
            yield from [i] * n

    for atom, increment in zip(top.atoms, increment_every_n(15)):
        atom.metadata["atom_serial"] += increment

    write_pdb_and_json(top, out_fn_template)


def discontiguous_resseq(top: Topology, out_fn_template: str):
    top = Topology(top)
    top = Topology(top)

    def increment_every_n(n: int):
        for i in count():
            yield from [i] * n

    for res, increment in zip(top.residues, increment_every_n(3)):
        for atom in res.atoms:
            atom.metadata["residue_number"] = (
                int(atom.metadata["residue_number"]) + increment
            )

    write_pdb_and_json(top, out_fn_template)


if __name__ == "__main__":
    main()
