import logging
import warnings
from collections.abc import Iterable, Mapping
from io import TextIOBase
from os import PathLike
from pathlib import Path
from typing import IO, Literal

import numpy as np
from openff.toolkit import Molecule, Topology
from openff.units import unit

from openff.pablo._matching import SuccessfulMatch

from ._pdb_data import PdbData
from ._utils import (
    cryst_to_box_vectors,
    sort_tuple,
)
from .ccd import STD_CCD_CACHE
from .residue import ResidueDefinition

__all__ = [
    "topology_from_pdb",
]


def topology_from_pdb(
    file: PathLike[str] | str | IO[str] | TextIOBase,
    *,
    residue_library: Mapping[
        str,
        Iterable[ResidueDefinition],
    ] = STD_CCD_CACHE,
    additional_definitions: Iterable[ResidueDefinition] = [],
    format: Literal["PDB", "CIF", None] = None,
    use_canonical_names: bool = False,
    ignore_unknown_CONECT_records: bool = False,
    set_stereochemistry_from_3d: bool = True,
) -> Topology:
    """
    Load a PDB file into an OpenFF ``Topology``.

    This function requires all hydrogens (and all other atoms) to be present in
    the PDB file, and that atom and residue names are consistent with the
    ``residue_library``. In return, it provides full chemical information on
    the entire PDB file.

    To load a PDB file with molecules including any residue not found in the
    CCD, or with residues that differ from that specified under a particular
    residue name, provide your own ``residue_library``. Any mapping from a
    residue name to a list of :py:data:`ResidueDefinition
    <openff.pdbscan.pdb.residue.ResidueDefinition>` objects may be used,
    but the :py:mod:`ccd <openff.pdbscan.pdb.ccd>` module  provides tools for
    augmenting the CCD.

    Alternatively, to load a single-residue molecule that is not present in the
    CCD, name that molecule ``"UNL"`` (or any name not present in the
    ``residue_library``), specify its CONECT records, and provide the
    appropriate molecule to the ``additional_definitions`` argument.

    Parameters
    ----------
    file
        The path to the PDB file or the PDB file as a file-like object.
    residue_library
        The database of residues to identify the atoms in the PDB file by. By
        default, a patched version of the CCD. Chemistry is identified by atom
        and residue names. If multiple residue definitions match a particular
        residue, the first one encountered is applied.
    additional_definitions
        Additional residue definitions to match against all residues that found
        no matches in the ``residue_library``. These definitions can match
        whether or not the residue name matches. Unlike ``residue_library``,
        this requires that CONECT records be present for any bonds not covered
        by the library and performs a match between the chemical graphs rather
        than using residue and atom names to detect chemistry. To use this
        argument with OpenFF ``Molecule`` objects or SMILES strings, see the
        ``ResidueDefinition.from_*`` class methods.
    format
        The file format the file is encoded in. `"PDB"` expects a standard PDB
        file. `"CIF"` expects a PDBx/mmCIF file. Omitting the argument or `None`
        interprets files with a known filename according to their filename
        extension, or an unknown filename extension as PDB.
    use_canonical_names
        If ``True``, atom names in the PDB file will be replaced by the
        canonical name for the same atom from the residue database.
    ignore_unknown_CONECT_records
        CONECT records do not include chemical information such as bond order
        and cannot be used on their own to add bonds beyond those specified
        through the residue database and unknown molecules. By default, any
        CONECT records not reflected in the final topology raise an error.
        If this argument is ``True``, this error is suppressed.
    set_stereochemistry_from_3d
        If ``True``, stereochemistry will be set according to the structure of
        the PDB file. If ``False``, leave stereo as set in the
        ``ResidueDefinition``.

    Notes
    -----

    This function uses a residue database to load a PDB file from its atom and
    residue names without guessing bonds. Bonds will be added by comparing atom
    and residue names to the residues defined in the ``residue_library``
    argument, which by default uses a patched version of the RCSB Chemical
    Component Dictionary (CCD). This is the dictionary of residue and atom names
    that the RCSB PDB is referenced against. The CCD is very large and cannot be
    distributed with this software, so by default internet access is required to
    use it.

    The produced ``Topology`` will have its atoms in the same order as the PDB
    file in all cases except when the atoms in one molecule are divided by
    another molecule. This can happen, for example, if a PDB file with 3 chains
    A, B and C has a disulfide bond between A and C. In this case, chains A and
    C form a single molecule, but the atoms from B should be in the middle. This
    atom ordering cannot be represented in :py:class:`openff.toolkit.Topology`
    unless all 3 chains are included in a single
    :py:class:`openff.toolkit.Molecule`, which would then represent two distinct
    chemical molecules. When this occurs, atoms from the latter chain(s) appear
    immediately after the first, and atoms from other molecules appear later.

    The following metadata are specified for all atoms produced by this function
    and can be accessed via ``topology.atom(i).metadata[key]``:

    ``"residue_name"``
        The residue name
    ``"residue_number"``
        The residue number, converted to an ``int``. If the residue number
        cannot be converted to an ``int``, the residue index instead.
    ``"res_seq"``
        The residue number as the string found in the PDB file
    ``"residue_index"``
        The residue index; the first residue has index ``0``, the second ``1``,
        etc., regardless of the value of the res_seq column.
    ``"insertion_code"``
        The icode for the atom's residue. Used to align residue numbers between
        proteins with indels.
    ``"chain_id"``
        The letter identifier for the atom's chain.
    ``"pdb_index"``
        The atom's index in the PDB file. Sometimes called rank. Not to be
        confused with ``"atom_serial"``, which is the number given to the atom
        in the second column of the PDB file. Guaranteed to be unique and to
        match the index of the atom within the topology.
    ``"used_synonym"``
        The name of the atom that was found in the PDB file. By default,
        `atom.name` is set to this. This value is not set for atoms matched via
        ``unknown_molecules``.
    ``"canonical_name"``
        The canonical name of the atom in the residue database. `atom.name` can
        be set to this with the `use_canonical_names` argument. This value is
        not set for atoms matched via ``unknown_molecules``.
    ``"atom_serial"``
        The serial number of the atom, found in the second column of the PDB
        file, as a string. Not guaranteed to be unique.
    ``"matched_residue_description"``
        The residue description found in the residue database. This value is not
        set for atoms matched via ``unknown_molecules``.
    ``"b_factor"``
        The temperature b-factor for the atom.
    ``"occupancy"``
        The occupancy for the atom.
    ``"alt_loc"``
        The alternate location code for the atom.
    ``"pdb_line_no"``
        The line number in the PDB file that contained this atom record.
    """
    if hasattr(file, "readlines"):
        if format is None and hasattr(file, "filename"):
            format = Path(file.name).suffix[1:].upper()  # type: ignore
        elif format is None or format == "":  # type: ignore
            format = "PDB"
        data = PdbData.from_file_object(file, format=format)  # type: ignore
    else:
        data = PdbData.from_file(file, format=format)  # type: ignore

    matches = data.get_successful_matches(
        residue_library,
        list(additional_definitions),
    )

    topology = _build_topology(
        matches=matches,
        data=data,
        use_canonical_names=use_canonical_names,
        set_stereochemistry_from_3d=set_stereochemistry_from_3d,
    )

    if not ignore_unknown_CONECT_records:
        _check_all_conects(topology, data)

    return topology


def _build_topology(
    matches: Iterable[SuccessfulMatch],
    data: PdbData,
    *,
    use_canonical_names: bool,
    set_stereochemistry_from_3d: bool,
) -> Topology:
    rdmol = data.matches_to_rdmol(matches, use_canonical_names=use_canonical_names)

    # Set positions
    logging.debug("Setting conformer to PDB positions")
    rdmol_pdb_indices = [atom.properties["pdb_index"] for atom in rdmol.atoms]
    positions = np.stack([data.x, data.y, data.z], axis=-1) * unit.angstrom
    rdmol = rdmol.edit().add_conformer_and(positions[rdmol_pdb_indices]).freeze()

    molecules: list[Molecule] = []
    for rdmol in rdmol.split_molecule_fragments():
        if set_stereochemistry_from_3d:
            rdmol = rdmol.edit().assign_stereochemistry_from_3d_and().freeze()
        offmol = rdmol.to_openff_molecule()
        offmol.add_default_hierarchy_schemes()
        molecules.append(offmol)

    logging.debug("produce topology")
    topology = Topology.from_molecules(molecules)

    topology_pdb_indices = [atom.metadata["pdb_index"] for atom in topology.atoms]
    if topology_pdb_indices != list(range(len(topology_pdb_indices))):
        logging.debug(
            "\n".join(
                f"topology index {j: <7} has pdb index {i: <7}"
                for j, i in enumerate(topology_pdb_indices)
                if i != j
            ),
        )
        warnings.warn(
            "Input PDB has an atom ordering that cannot be represented in an"
            + " OpenFF Topology. The atoms in this topology will not be in same"
            + " order as those in PDB file",
        )

    _set_box_vectors(topology, data)

    return topology


def _check_all_conects(topology: Topology, data: PdbData):
    all_bonds: set[tuple[int, int]] = {
        sort_tuple((bond.atom1.metadata["pdb_index"], bond.atom2.metadata["pdb_index"]))  # type:ignore
        for bond in topology.bonds
    }

    conect_bonds: set[tuple[int, int]] = set()
    for i, js in enumerate(data.conects):
        for j in js:
            conect_bonds.add(sort_tuple((i, j)))
    if not conect_bonds.issubset(all_bonds):
        raise ValueError(
            "CONECT records without chemical information not supported",
            sorted(
                {
                    sort_tuple((data.serial[a], data.serial[b]))
                    for a, b in conect_bonds.difference(all_bonds)
                },
            ),
        )


def _set_box_vectors(topology: Topology, data: PdbData):
    if (
        data.cryst1_a is not None
        and data.cryst1_b is not None
        and data.cryst1_c is not None
        and data.cryst1_alpha is not None
        and data.cryst1_beta is not None
        and data.cryst1_gamma is not None
    ):
        topology.box_vectors = cryst_to_box_vectors(
            data.cryst1_a,
            data.cryst1_b,
            data.cryst1_c,
            data.cryst1_alpha,
            data.cryst1_beta,
            data.cryst1_gamma,
        )
