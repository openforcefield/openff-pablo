from io import StringIO
from itertools import count
from pathlib import Path
from collections.abc import Callable, Iterable

from openff.toolkit import Topology

from openff.pablo._utils import T


def main():
    in_fn = Path(__file__).parent / "2MUM_neutralized.json"
    out_fn_template = str(Path(__file__).parent / "2MUM_{}")

    topology = Topology.from_json(in_fn.read_text())

    funcs = [
        reuse_serial,
        reuse_resseq,
        icode,
        discontiguous_serial,
        letters_in_serial,
        letters_in_resseq,
    ]

    for func in [
        dryrun,
        *funcs,
        compose(funcs),
    ]:
        top = Topology(topology)
        func(top)
        write_pdb_and_json(top, out_fn_template.format(func.__name__))


def compose(funcs: Iterable[Callable[[T], T]]) -> Callable[[T], T]:
    def composed_function(arg):
        value = arg
        for func in funcs:
            value = func(value)
        return value

    return composed_function


def dryrun(top: Topology):
    return top


def write_pdb_and_json(top: Topology, out_fn_stem: str):
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

    Path(out_fn_stem).with_suffix(".pdb").write_text("\n".join(lines))
    Path(out_fn_stem).with_suffix(".json").write_text(top.to_json())


def reuse_serial(top: Topology) -> Topology:
    res_iter = iter(top.residues)
    for i in range(5):
        reused_serial = next(res_iter).atom(-1).metadata["atom_serial"]
    for res in res_iter:
        for atom in res.atoms:
            atom.metadata["atom_serial"] = reused_serial

    return top


def reuse_resseq(top: Topology) -> Topology:
    res_iter = iter(top.residues)
    for i in range(5):
        res = next(res_iter)
        reused_resseq = res.identifier[1]
        prev_resname = res.identifier[3]
    for res in res_iter:
        resname = res.identifier[3]
        icode = res.identifier[2]
        if resname == prev_resname:
            icode = "A" if icode == " " else chr(ord(icode) + 1)
        for atom in res.atoms:
            atom.metadata["residue_number"] = reused_resseq
            atom.metadata["insertion_code"] = icode
        prev_resname = resname

    return top


def letters_in_serial(top: Topology) -> Topology:
    for atom in top.atoms:
        serial = str(atom.metadata["atom_serial"])
        assert len(serial) <= 4
        atom.metadata["atom_serial"] = f"A{serial:0>4}"

    return top


def letters_in_resseq(top: Topology) -> Topology:
    for atom in top.atoms:
        resseq = atom.metadata["residue_number"]
        assert len(str(resseq)) <= 3
        atom.metadata["residue_number"] = f"A{resseq:0>3}"

    return top


def icode(top: Topology) -> Topology:
    prev_resname = None
    resseq_offset = 0
    icode_offset = 0
    for i, res in enumerate(top.residues, start=1):
        resname = res.identifier[3]
        if resname == prev_resname or i % 15 == 0 or (i % 30 == 1 and i > 15):
            resseq_offset += 1
            icode_offset += 1
        else:
            icode_offset = 0
        for atom in res.atoms:
            atom.metadata["residue_number"] = (
                int(atom.metadata["residue_number"]) - resseq_offset
            )
            if icode_offset >= 1:
                icode = atom.metadata["insertion_code"]
                icode_ord = ord("A") - 1 if icode == " " else ord(icode)
                atom.metadata["insertion_code"] = chr(icode_ord + icode_offset)
        prev_resname = resname

    return top


def increment_every_n(n: int):
    for i in count():
        yield from [i] * n


def discontiguous_serial(top: Topology) -> Topology:
    for atom, increment in zip(top.atoms, increment_every_n(15)):
        atom.metadata["atom_serial"] += increment

    return top


def discontiguous_resseq(top: Topology) -> Topology:
    for res, increment in zip(top.residues, increment_every_n(3)):
        for atom in res.atoms:
            atom.metadata["residue_number"] = (
                int(atom.metadata["residue_number"]) + increment
            )

    return top


if __name__ == "__main__":
    main()
