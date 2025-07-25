"""This generates a PDB file including every kind of residue-residue interface
understood by ``PdbData.filter_on_polymer_linkages()``:

- no_polymer_new_chain (water chain B -> C)
- no_polymer_after_ter (water chain C -> X and X -> X)
- no_polymer_no_links (water chain C -> C)
- continue_polymer (middle glycine residue of all triglycine molecules)
- end_polymer_with_new_chain (triglycine chain A -> B)
- end_polymer_with_ter (triglycine chain B -> B)
- end_polymer_no_links (triglycine chain B -> water chain B)
- start_polymer_with_new_chain (triglycine chain A -> B)
- start_polymer_with_ter (triglycine chain B -> B)
- start_polymer_no_links (triglycine chain B -> water chain B)

Note that the triglycines begin and end with a negatively charged amine and a
carboxylic carbanion so that adjacent glycine chains have the right atoms to
form a polymer bond. For this to load, two special glycine residue definitions
are needed:

ResidueDefinition.from_smiles(
    mapped_smiles="[N-:1]([H:2])[C:3]([H:4])([H:5])[C:6](=[O:7])[O:8][H:9]",
    atom_names={1: "N", 2: "H", 3: "CA", 4: "HA1", 5: "HA2", 6: "C", 7: "O", 8: "OXT", 9: "HXT"},
    residue_name="GLY",
    leaving_atoms=(8, 9),
    linking_bond=PEPTIDE_BOND,
    description="GLYCINE w/ negative formal charge on N",
)
ResidueDefinition.from_smiles(
    mapped_smiles="[N:1]([H:2])([H:8])[C:3]([H:4])([H:5])[C-:6]=[O:7]",
    atom_names={1: "N", 2: "H", 3: "CA", 4: "HA1", 5: "HA2", 6: "C", 7: "O", 8: "H2"},
    residue_name="GLY",
    leaving_atoms=(8,),
    linking_bond=PEPTIDE_BOND,
    description="GLYCINE w/ negative formal charge on C",
)

"""

from io import StringIO
from pathlib import Path
from collections.abc import Iterable

from openff.toolkit import Molecule, Topology
from openff.units import Quantity

from openff.interchange.components._packmol import pack_box


def molecule_from_smiles_and_metadata(
    smiles: str,
    metadata_dict: dict[Iterable[int], dict[str, str]],
) -> Molecule:
    molecule = Molecule.from_mapped_smiles(smiles)
    for maps, metadata in metadata_dict.items():
        for m in maps:
            atom = molecule.atom(m - 1)
            for key, value in metadata.items():
                if key == "atom_name":
                    atom.name = value
                else:
                    atom.metadata[key] = value

    return molecule


def triglycine(chain_id: str = "X", terminated: bool = True) -> Molecule:
    return molecule_from_smiles_and_metadata(
        smiles=(
            "[N-:1]([H:2])[C:3]([H:4])([H:5])[C:6](=[O:7])"
            + "[N:8]([H:9])[C:10]([H:11])([H:12])[C:13](=[O:14])"
            + "[N:15]([H:16])[C:17]([H:18])([H:19])[C-:20](=[O:21])"
        ),
        metadata_dict={
            range(1, 22): {"residue_name": "GLY", "chain_id": chain_id},
            range(1, 8): {"residue_number": "1"},
            range(8, 15): {"residue_number": "2"},
            range(15, 22): {"residue_number": "3"},
            (1, 8, 15): {"atom_name": "N"},
            (2, 9, 16): {"atom_name": "H"},
            (3, 10, 17): {"atom_name": "CA"},
            (4, 11, 18): {"atom_name": "HA1"},
            (5, 12, 19): {"atom_name": "HA2"},
            (6, 13, 20): {"atom_name": "C"},
            (7, 14, 21): {"atom_name": "O"},
            (21,): {"terminated": terminated},
        },
    )


def water(chain_id: str = "X", terminated: bool = True) -> Molecule:
    return molecule_from_smiles_and_metadata(
        smiles="[H:2][O:1][H:3]",
        metadata_dict={
            range(1, 4): {
                "residue_name": "HOH",
                "chain_id": chain_id,
                "residue_number": "1",
            },
            (1,): {"atom_name": "O"},
            (2,): {"atom_name": "H1"},
            (3,): {
                "atom_name": "H2",
                "terminated": terminated,
            },
        },
    )


def write_pdb_from_metadata(filename: Path | str, topology: Topology):
    path = Path(filename)
    lines = []
    for serial, (atom, xyz) in enumerate(zip(topology.atoms, topology.get_positions())):
        name = atom.name
        altloc = atom.metadata.get("alt_loc", " ")
        resname = atom.metadata.get("residue_name", None)
        resseq = atom.metadata.get("residue_number", None)
        chainid = atom.metadata.get("chain_id", None)
        icode = atom.metadata.get("insertion_code", " ")
        bfac = atom.metadata.get("b_factor", "0.0")
        occ = atom.metadata.get("occupancy", "1.0")
        x, y, z = xyz.m_as("nm")
        elem = atom.symbol

        assert all(
            [
                name,
                resname,
                resseq,
                chainid,
                len(name) <= 5,
                len(resname) <= 4,
                len(chainid) == 1,
                len(icode) == 1,
                len(altloc) == 1,
                len(elem) <= 2,
                serial < 100000,
                len(str(resseq)) <= 5,
            ],
        )
        lines.append(
            f"ATOM  {serial: >5} {name: >4}{altloc}{resname: <4}{chainid}"
            + f"{resseq: >4}{icode}   {x:6.2f}{y:6.2f}{z:6.2f}{occ:1.3}"
            + f"{bfac:1.3}          {elem: <2}",
        )
        if atom.metadata.get("terminated", False):
            lines.append("TER")

    path.write_text("\n".join(lines))


def main():
    topology: Topology = pack_box(
        molecules=[
            triglycine("A", terminated=False),
            triglycine("B", terminated=True),
            triglycine("B", terminated=False),
            water("B", terminated=False),
            water("B", terminated=False),
            water("C", terminated=True),
            water(),
        ],
        number_of_copies=[1, 1, 1, 1, 1, 1, 97],
        target_density=Quantity("0.8 g/ml"),
    )
    # write_pdb_from_metadata(Path(__file__).with_suffix(".pdb"), topology)

    with StringIO() as f:
        topology.to_file(
            f,
            file_format="PDB",
            keep_ids=True,
            ensure_unique_atom_names=False,
        )
        pdbstr = f.getvalue()

    atoms = iter(topology.atoms)
    remove_ters = False
    pdblines = []
    for line in pdbstr.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            atom = next(atoms)
            remove_ters = not atom.metadata.get("terminated", False)
        if line.startswith("TER   ") and remove_ters:
            continue
        pdblines.append(line)

    Path(__file__).with_suffix(".pdb").write_text("\n".join(pdblines))


if __name__ == "__main__":
    main()
