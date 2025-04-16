import itertools
import warnings
from collections.abc import Iterable, Mapping, MutableSequence
from io import TextIOBase
from os import PathLike
from typing import IO, assert_never

import numpy as np
from openff.toolkit import Molecule, Topology
from openff.units import elements, unit

from ._pdb_data import PdbData, ResidueMatch
from ._utils import (
    assign_stereochemistry_from_3d,
    cryst_to_box_vectors,
    sort_tuple,
)
from .ccd import CCD_RESIDUE_DEFINITION_CACHE
from .exceptions import (
    MultipleMatchingResidueDefinitionsError,
    NoMatchingResidueDefinitionError,
)
from .residue import ResidueDefinition

__all__ = [
    "topology_from_pdb",
]


def _match_unknown_molecules(
    data: PdbData,
    indices: tuple[int, ...],
    unknown_molecules: Iterable[Molecule],
) -> Molecule | None:
    conects: set[tuple[int, int]] = set()
    pdb_idx_to_mol_idx: dict[int, int] = {}
    pdbmol = Molecule()
    for pdb_index in indices:
        pdb_idx_to_mol_idx[pdb_index] = pdbmol.add_atom(
            atomic_number=elements.NUMBERS[data.element[pdb_index]],
            formal_charge=data.charge[pdb_index] or 0,
            is_aromatic=False,
            stereochemistry=None,
            name=data.name[pdb_index],
            metadata={
                "residue_name": data.res_name[pdb_index],
                "leaving": False,
                "pdb_index": pdb_index,
                "residue_number": str(data.res_seq[pdb_index]),
                "res_seq": data.res_seq[pdb_index],
                "insertion_code": data.i_code[pdb_index],
                "chain_id": data.chain_id[pdb_index],
                "atom_serial": data.serial[pdb_index],
                "b_factor": str(data.temp_factor[pdb_index]),
                "occupancy": str(data.occupancy[pdb_index]),
                "alt_loc": str(data.alt_loc[pdb_index]),
            },
        )
        for conect_idx in data.conects[pdb_index]:
            conects.add(sort_tuple((pdb_index, conect_idx)))

    for a, b in conects:
        try:
            pdbmol.add_bond(
                atom1=pdb_idx_to_mol_idx[a],
                atom2=pdb_idx_to_mol_idx[b],
                bond_order=1,
                is_aromatic=False,
            )
        except KeyError:
            a_summary = f"{data.name[a]}#{data.serial[a]}@{data.chain_id[a]}:{data.res_name[a]}#{data.res_seq[a]}"
            b_summary = f"{data.name[b]}#{data.serial[b]}@{data.chain_id[b]}:{data.res_name[b]}#{data.res_seq[b]}"
            raise ValueError(
                "Cannot match unknown molecule that spans multiple residues: "
                + f"Found CONECT record between {a_summary} and {b_summary}",
            )

    for molecule in unknown_molecules:
        (match_found, mapping) = Molecule.are_isomorphic(
            molecule,
            pdbmol,
            return_atom_map=True,
            aromatic_matching=False,
            formal_charge_matching=False,
            bond_order_matching=False,
            atom_stereochemistry_matching=False,
            bond_stereochemistry_matching=False,
            strip_pyrimidal_n_atom_stereo=True,
        )
        if match_found:
            assert mapping is not None
            molecule = molecule.remap(mapping)
            for atom, pdbatom in zip(molecule.atoms, pdbmol.atoms):
                atom.metadata.update(pdbatom.metadata)
                atom.name = pdbatom.name
            molecule.generate_conformers(n_conformers=0, clear_existing=True)
            molecule.properties["pdb_idx_to_mol_atom_idx"] = pdb_idx_to_mol_idx

            return molecule
    else:
        return None


def topology_from_pdb(
    file: PathLike[str] | str | IO[str] | TextIOBase,
    *,
    unknown_molecules: Iterable[Molecule] = [],
    residue_database: Mapping[
        str,
        Iterable[ResidueDefinition],
    ] = CCD_RESIDUE_DEFINITION_CACHE,
    additional_substructures: Iterable[ResidueDefinition] = [],
    use_canonical_names: bool = False,
    ignore_unknown_CONECT_records: bool = False,
    set_stereochemistry_from_3d: bool = True,
    verbose_errors: bool = False,
) -> Topology:
    """
    Load a PDB file into an OpenFF ``Topology``.

    This function requires all hydrogens (and all other atoms) to be present in
    the PDB file, and that atom and residue names are consistent with the
    ``residue_database``. In return, it provides full chemical information on
    the entire PDB file.

    To load a PDB file with molecules including any residue not found in the CCD,
    or with residues that differ from that specified under a particular residue
    name, provide your own `residue_database`. Any mapping from a residue
    name to a list of [`ResidueDefinition`][openff.pablo.ResidueDefinition]
    objects may be used, but the [`openff.pablo.ccd`][] module provides tools
    for augmenting the CCD.

    Alternatively, to load a single-residue molecule that is not present in the
    CCD, name that molecule `"UNL"` (or any name not present in the
    `residue_database`), specify its CONECT records, and provide the appropriate
    molecule to the ``unknown_molecules`` argument.

    Parameters
    ----------
    file
        The path to the PDB file or the PDB file as a file-like object.
    unknown_molecules
        A list of molecules to match residues not found in the
        ``residue_database`` against. Unlike ``residue_database``, this requires
        that CONECT records be present and performs a match between the chemical
        graphs rather than using residue and atom names to detect chemistry.
    residue_database
        The database of residues to identify the atoms in the PDB file by. By
        default, a patched version of the CCD. Chemistry is identified by atom
        and residue names. If multiple residue definitions match a particular
        residue, the first one encountered is applied.
    additional_substructures
        Additional residue definitions to match against all residues that found
        no matches in the ``residue_database``. These definitions can match
        whether or not the residue name matches. To use this argument with
        OpenFF ``Molecule`` objects or SMILES strings, see the
        ``ResidueDefinition.from_*`` class methods.
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
        the PDB file. This takes considerable time. If ``False``, leave stereo
        as set in the ``ResidueDefinition``.
    verbose_errors
        If ``True``, give more detailed error reports. These can get quite long.

    Notes
    -----

    This function uses a residue database to load a PDB file from its atom and
    residue names without guessing bonds. Bonds will be added by comparing atom
    and residue names to the residues defined in the ``residue_database``
    argument, which by default uses a patched version of the RCSB Chemical
    Component Dictionary (CCD). This is the dictionary of residue and atom names
    that the RCSB PDB is referenced against. The CCD is very large and cannot be
    distributed with this software, so by default except for a select few common
    residues internet access is required to use it.

    The produced `Topology` will have its atoms in the same order as the PDB
    file in all cases except when the atoms in one molecule are divided by
    another molecule. This can happen, for example, if a PDB file with 3 chains
    A, B and C has a disulfide bond between A and C. In this case, chains A and
    C form a single molecule, but the atoms from B should be in the middle. This
    atom ordering cannot be represented in a
    [`Topology`][openff.toolkit.topology.Topology] unless all 3 chains are
    included in a single [`Molecule`][openff.toolkit.topology.Molecule], which
    would then represent two distinct molecules. When this occurs, atoms from
    the latter chain(s) appear immediately after the first, and atoms from other
    molecules appear after.

    The following metadata are specified for all atoms produced by this function
    and can be accessed via `topology.atom(i).metadata[key]`:

    `"residue_name"`
    :   The residue name

    `"residue_number"`
    :   The residue number as a string

    `"res_seq"`
    :   The residue number as an integer

    `"insertion_code"`
    :   The icode for the atom's residue. Used to align residue numbers between
        proteins with indels.

    `"chain_id"`
    :   The letter identifier for the atom's chain.

    `"pdb_index"`
    :   The atom's index in the PDB file. Sometimes called rank. Not to be
        confused with ``"atom_serial"``, which is the number given to the atom
        in the second column of the PDB file. Guaranteed to be unique and to
        match the index of the atom within the topology.

    `"used_synonym"`
    :   The name of the atom that was found in the PDB file. By default,
        `atom.name` is set to this. This value is not set for atoms matched via
        ``unknown_molecules``.

    `"canonical_name"`
    :   The canonical name of the atom in the residue database. `atom.name` can
        be set to this with the `use_canonical_names` argument. This value is
        not set for atoms matched via ``unknown_molecules``.

    `"atom_serial"`
    :   The serial number of the atom, found in the second column of the PDB
        file. Not guaranteed to be unique.

    `"matched_residue_description"`
    :   The residue description found in the residue database. This value is not
        set for atoms matched via ``unknown_molecules``.

    `"b_factor"`
    :   The temperature b-factor for the atom.

    `"occupancy"`
    :   The occupancy for the atom.

    `"alt_loc"`
    :   The alternate location code for the atom.

    """
    if hasattr(file, "readlines"):
        data = PdbData.from_file_object(file)  # type: ignore
    else:
        data = PdbData.from_file(file)  # type: ignore

    this_molecule = Molecule()
    molecules: list[Molecule] = [this_molecule]
    prev_chain_id = data.chain_id[0]
    prev_model = data.model[0]
    for res_atom_idcs, matches in zip(
        data.residue_indices,
        data.get_residue_matches(residue_database, additional_substructures),
    ):
        # Check that we have a unique match, and error out or consult
        # unique_molecules as appropriate
        chemical_data: Molecule | ResidueMatch
        if len(matches) == 0:
            unknown_molecule = _match_unknown_molecules(
                data,
                res_atom_idcs,
                unknown_molecules,
            )
            if unknown_molecule is None:
                raise NoMatchingResidueDefinitionError(
                    res_atom_idcs,
                    data,
                    unknown_molecules,
                    additional_substructures,
                    residue_database,
                    verbose_errors=verbose_errors,
                )
            else:
                chemical_data = unknown_molecule
        # If all matches would assign the same chemistry, accept it
        # assert all([]) == True
        elif all(a.agrees_with(b) for a, b in itertools.pairwise(matches)):
            chemical_data = matches[0]

            # this is a debug assert, if it triggers there's a bug
            assert set(res_atom_idcs) == chemical_data.res_atom_idcs
        else:
            raise MultipleMatchingResidueDefinitionsError(
                matches,
                res_atom_idcs,
                data,
                verbose_errors=verbose_errors,
            )

        prototype_index = res_atom_idcs[0]

        # Terminate the previous molecule and start a new one if we can see that
        # this is the start of a new molecule
        if this_molecule.n_atoms > 0 and (
            data.chain_id[prototype_index] != prev_chain_id
            or data.model[prototype_index] != prev_model
            or (
                isinstance(chemical_data, ResidueMatch)
                and not chemical_data.expect_prior_bond
            )
        ):
            this_molecule = Molecule()
            molecules.append(this_molecule)

        # Apply the chemical data we've collected
        if isinstance(chemical_data, Molecule):
            this_molecule = chemical_data
            if molecules[-1].n_atoms == 0:
                molecules[-1] = this_molecule
            else:
                molecules.append(this_molecule)
        elif isinstance(chemical_data, ResidueMatch):
            this_molecule = _add_to_molecule(
                molecules,
                this_molecule,
                chemical_data,
                data,
                use_canonical_names,
            )
        else:
            assert_never(chemical_data)

        # Terminate the current molecule if we can see that this is the last residue
        if (
            data.terminated[prototype_index]
            or isinstance(chemical_data, Molecule)
            or (
                isinstance(chemical_data, ResidueMatch)
                and not chemical_data.expect_posterior_bond
            )
        ):
            this_molecule = Molecule()
            molecules.append(this_molecule)

        # TODO: Load other data from PDB file
        # TODO: Incorporate CONECT records
        # TODO: Deal with multi-model files

        prev_chain_id = data.chain_id[prototype_index]
        prev_model = data.model[prototype_index]

    for offmol in molecules:
        offmol._invalidate_cached_properties()
        offmol.add_default_hierarchy_schemes()

    topology = Topology.from_molecules(filter(lambda m: m.n_atoms != 0, molecules))

    topology_pdb_indices = [atom.metadata["pdb_index"] for atom in topology.atoms]
    n = len(topology_pdb_indices)
    positions = np.stack([data.x[:n], data.y[:n], data.z[:n]], axis=-1) * unit.angstrom
    topology.set_positions(positions[topology_pdb_indices])
    if topology_pdb_indices != list(range(n)):
        warnings.warn(
            "Input PDB has an atom ordering that cannot be represented in an"
            + " OpenFF Topology. The atoms in this topology will not be in same"
            + " order as those in PDB file",
        )

    if set_stereochemistry_from_3d:
        for molecule in topology.molecules:
            # TODO: Speed this up
            #   - Build up molecules in RDMol form to skip conversion step?
            # This accounts for nearly half of the time to load 5ap1_prepared.pdb
            assign_stereochemistry_from_3d(molecule)

    if not ignore_unknown_CONECT_records:
        _check_all_conects(topology, data)

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


def _add_to_molecule(
    molecules: MutableSequence[Molecule],
    this_molecule: Molecule,
    residue_match: ResidueMatch,
    data: PdbData,
    use_canonical_names: bool,
) -> Molecule:
    # Identify the previous linking atom
    linking_atom_idx: None | int = None
    if residue_match.expect_prior_bond:
        assert residue_match.residue_definition.linking_bond is not None
        linking_atom_name = residue_match.residue_definition.linking_bond.atom1
        for i in reversed(range(this_molecule.n_atoms)):
            if this_molecule.atom(i).metadata["canonical_name"] == linking_atom_name:
                linking_atom_idx = i
                break
        assert linking_atom_idx is not None, (
            "Expecting a prior bond, but no linking atom found"
        )

    # Add the residue to the current molecule
    atom_name_to_mol_idx: dict[str, int] = {}
    pdb_idx_to_mol_idx: dict[int, int] = this_molecule.properties.setdefault(
        "pdb_idx_to_mol_atom_idx",
        {},
    )
    for pdb_index in sorted(residue_match.res_atom_idcs):
        atom_def = residue_match.atom(pdb_index)

        mol_atom_idx = this_molecule._add_atom(
            atomic_number=elements.NUMBERS[atom_def.symbol],
            formal_charge=atom_def.charge,
            is_aromatic=atom_def.aromatic,
            stereochemistry=None,
            name=atom_def.name if use_canonical_names else data.name[pdb_index],
            metadata={
                "residue_name": data.res_name[pdb_index],
                "res_seq": data.res_seq[pdb_index],
                "residue_number": str(data.res_seq[pdb_index]),
                "insertion_code": data.i_code[pdb_index],
                "chain_id": data.chain_id[pdb_index],
                "pdb_index": pdb_index,
                "used_synonym": data.name[pdb_index],
                "canonical_name": atom_def.name,
                "atom_serial": data.serial[pdb_index],
                "matched_residue_description": residue_match.residue_definition.description,
                "b_factor": str(data.temp_factor[pdb_index]),
                "occupancy": str(data.occupancy[pdb_index]),
                "alt_loc": str(data.alt_loc[pdb_index]),
            },
            invalidate_cache=False,
        )
        atom_name_to_mol_idx[atom_def.name] = mol_atom_idx
        pdb_idx_to_mol_idx[pdb_index] = mol_atom_idx

    for bond in residue_match.residue_definition.bonds:
        if bond.atom1 in atom_name_to_mol_idx and bond.atom2 in atom_name_to_mol_idx:
            this_molecule._add_bond(
                atom1=atom_name_to_mol_idx[bond.atom1],
                atom2=atom_name_to_mol_idx[bond.atom2],
                bond_order=bond.order,
                is_aromatic=bond.aromatic,
                stereochemistry=bond.stereo,
                invalidate_cache=False,
            )

    if linking_atom_idx is not None:
        linking_bond = residue_match.residue_definition.linking_bond
        assert linking_bond is not None, (
            "linking_atom_idx is only set when linking_atom_idx is None"
        )
        this_molecule._add_bond(
            atom1=linking_atom_idx,
            atom2=atom_name_to_mol_idx[linking_bond.atom2],
            bond_order=linking_bond.order,
            is_aromatic=linking_bond.aromatic,
            stereochemistry=linking_bond.stereo,
            invalidate_cache=False,
        )

    if residue_match.crosslink is not None:
        this_idx, other_idx = residue_match.crosslink
        crosslink_bond = residue_match.residue_definition.crosslink
        assert crosslink_bond is not None, "Crosslink cannot be None if in match"
        if other_idx > this_idx:
            # If this residue is the first residue of the crosslink to be added,
            # skip it and wait for the other residue to be read.
            return this_molecule

        # If the crosslink is within this molecule, just add the bond
        if other_idx in pdb_idx_to_mol_idx:
            this_molecule._add_bond(
                atom1=pdb_idx_to_mol_idx[this_idx],
                atom2=pdb_idx_to_mol_idx[other_idx],
                bond_order=crosslink_bond.order,
                is_aromatic=crosslink_bond.aromatic,
                stereochemistry=crosslink_bond.stereo,
                invalidate_cache=False,
            )
            return this_molecule

        for other_molecule in molecules:
            other_mol_pdb_idx_to_mol_atom_idx: dict[int, int] = (
                other_molecule.properties["pdb_idx_to_mol_atom_idx"]
            )
            assert isinstance(
                other_mol_pdb_idx_to_mol_atom_idx,
                dict,
            ), "This property should have already been set by Pablo"

            if other_idx in other_mol_pdb_idx_to_mol_atom_idx:
                # Forming a crosslink to a previously terminated molecule
                # Transfer all atoms from this molecule into the other
                old_to_new: dict[int, int] = {}
                for old_idx, atom in enumerate(this_molecule.atoms):
                    old_to_new[old_idx] = other_molecule._add_atom(
                        atomic_number=atom.atomic_number,
                        formal_charge=atom.formal_charge.m,  # type:ignore
                        is_aromatic=atom.is_aromatic,
                        stereochemistry=atom.stereochemistry,
                        name=atom.name,
                        metadata=dict(atom.metadata),
                        invalidate_cache=False,
                    )
                for bond in this_molecule.bonds:
                    other_molecule._add_bond(
                        atom1=old_to_new[bond.atom1_index],
                        atom2=old_to_new[bond.atom2_index],
                        bond_order=bond.bond_order,
                        is_aromatic=bond.is_aromatic,
                        stereochemistry=bond.stereochemistry,
                        invalidate_cache=False,
                    )
                other_mol_pdb_idx_to_mol_atom_idx.update(
                    {k: old_to_new[v] for k, v in pdb_idx_to_mol_idx.items()},
                )
                # Add the crosslink
                other_molecule._add_bond(
                    atom1=other_mol_pdb_idx_to_mol_atom_idx[this_idx],
                    atom2=other_mol_pdb_idx_to_mol_atom_idx[other_idx],
                    bond_order=crosslink_bond.order,
                    is_aromatic=crosslink_bond.aromatic,
                    stereochemistry=crosslink_bond.stereo,
                    invalidate_cache=False,
                )
                # Discard the old molecule
                molecules[:] = [mol for mol in molecules if mol is not this_molecule]
                return other_molecule

    return this_molecule
