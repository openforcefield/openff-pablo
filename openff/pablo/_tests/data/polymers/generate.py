import json
from pathlib import Path

from openff.toolkit import Topology
from openff.toolkit.utils.exceptions import (
    PDBMoleculeHasNoncontiguousAtomIndicesError,
    SubstructureAtomSmartsInvalid,
    UnassignedChemistryInPDBError,
)


def successfully_loaded(top: Topology) -> bool:
    match_info = [atom.metadata["match_info"] for atom in top.atoms]
    return all([bool(match) for match in match_info])


if __name__ == "__main__":
    for dir in Path(__file__).parent.iterdir():
        if not dir.is_dir():
            continue
        for pdb_file in dir.iterdir():
            if not pdb_file.suffix.endswith(".pdb"):
                continue
            print(pdb_file)
            monomers_file_out = pdb_file.with_suffix(".monomers.json")
            monomers_file_in = Path(
                "../polymer_examples/monomer_generation/json_files",
                pdb_file.with_suffix(".json").name,
            )
            topology_file_out = pdb_file.with_suffix(".topology.json")
            if topology_file_out.exists():
                print("  skipping (topology already exists)")
                continue

            try:
                substructs = json.loads(monomers_file_in.read_text())["monomers"]
            except FileNotFoundError:
                substructs = {}

            try:
                top = Topology.from_pdb(pdb_file, _custom_substructures=substructs)
                assert successfully_loaded(top)
            except (
                PDBMoleculeHasNoncontiguousAtomIndicesError,
                UnassignedChemistryInPDBError,
                SubstructureAtomSmartsInvalid,
            ) as e:
                print(f"  Skipping because of error: {e.__class__.__name__}")
                continue

            topology_file_out.write_text(top.to_json())

            if substructs:
                monomers_file_out.write_text(monomers_file_in.read_text())
