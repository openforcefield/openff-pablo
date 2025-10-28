import dataclasses
import itertools
import logging
import warnings
from collections import defaultdict
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from io import IOBase, TextIOBase
from os import PathLike
from pathlib import Path
from typing import IO, Any, DefaultDict, Literal, Protocol, Self

from openff.pablo._additional_definitions import apply_additional_definitions
from openff.pablo._cif import (
    cif_floats,
    cif_ints,
    cif_opt_floats,
    cif_opt_ints,
    cif_strs,
    parse_cif,
)
from openff.pablo._exceptions import (
    PabloError,
    UnknownOrAmbiguousSerialInConectError,
    create_pdb_residue_match_error,
)
from openff.pablo._graph import Graph
from openff.pablo._matching import (
    MismatchProtocol,
    NoResidueDefinitions,
    PossibleResidueMatch,
    ResidueConectMatch,
    ResidueMatch,
    ResidueMismatch,
    SuccessfulMatch,
    only_matched,
)
from openff.pablo._rdkit import EditableRdMol, RdAtom, RdMol
from openff.pablo._utils import (
    __UNSET__,
    charge_int_or_none,
    dec_hex,
    flatten,
    no_none_in_values,
    sort_tuple,
    unwrap,
    unwrap_or_none,
)
from openff.pablo.residue import AtomDefinition, BondDefinition, ResidueDefinition

__all__ = [
    "PdbData",
]


@dataclass
class PdbData:
    src_filename: str | None = None
    line_no: list[int] = field(default_factory=list[int])
    model: list[int | None] = field(default_factory=list[int | None])
    serial: list[str] = field(default_factory=list[str])
    name: list[str] = field(default_factory=list[str])
    alt_loc: list[str] = field(default_factory=list[str])
    res_name: list[str] = field(default_factory=list[str])
    chain_id: list[str] = field(default_factory=list[str])
    res_seq: list[str] = field(default_factory=list[str])
    i_code: list[str] = field(default_factory=list[str])
    x: list[float] = field(default_factory=list[float])
    y: list[float] = field(default_factory=list[float])
    z: list[float] = field(default_factory=list[float])
    occupancy: list[float] = field(default_factory=list[float])
    temp_factor: list[float] = field(default_factory=list[float])
    element: list[str] = field(default_factory=list[str])
    charge: list[int | None] = field(default_factory=list[int | None])
    terminated: list[bool] = field(default_factory=list[bool])
    res_idx: list[int] | None = None
    serial_to_index: DefaultDict[str, list[int]] = field(
        default_factory=lambda: defaultdict(list),
    )
    conects: list[set[int]] = field(default_factory=list[set[int]])
    """The ith set contains atom indices CONECTed to atom index i"""
    cryst1_a: float | None = None
    cryst1_b: float | None = None
    cryst1_c: float | None = None
    cryst1_alpha: float | None = None
    cryst1_beta: float | None = None
    cryst1_gamma: float | None = None
    strict: bool = False
    """True to read in strict, spec-compliant mode, False to use common extensions.

    Extensions include:
    - hexadecimal extensions for atom serials and residue sequence numbers that
    do not fit in the fixed column format
    - missing charge column indicates unknown charge, not 0
    - residue name may extend into column 21 iff the reside name has 4
    non-whitespace characters"""
    _cursor: int = 0
    """Line number currently being read"""

    def __repr__(self) -> str:
        return (
            f"PdbData.from_file({self.src_filename})"
            if self.src_filename is not None
            else f"<PdbData with {len(self.name)} records>"
        )

    @classmethod
    def from_file(
        cls,
        path: str | PathLike[str],
        format: Literal["PDB", "CIF", None] = None,
    ) -> Self:
        """
        Create a ``PdbData`` object by reading from a file.

        Parameters
        ----------
        path
            The path to the PDB file
        format
            Which format to interpret the file as. (default: infer from file
            name extension)
        """
        path = Path(path)
        if format is None:
            format = "CIF" if path.suffix.lower() == ".cif" else "PDB"
        with open(path) as f:
            ret = cls.from_file_object(f, format=format)
        ret.src_filename = str(path)
        return ret

    @classmethod
    def from_file_object(
        cls,
        file: IO[str] | TextIOBase | IO[bytes] | IOBase,
        format: Literal["PDB", "CIF"] = "PDB",
    ) -> Self:
        """
        Create a ``PdbData`` object by reading from a file-like object.

        Parameters
        ----------
        file
            A file-like object containing PDB data
        format
            Which format to interpet the file as
        """
        lines = [
            (line if isinstance(line, str) else line.decode())
            for line in file.readlines()
        ]
        if format.upper() == "PDB":
            return cls.parse_pdb(lines)
        elif format.upper() == "CIF":
            return cls.parse_cif(lines)
        else:
            raise PabloError(f"format must be one of 'PDB' or 'CIF', not {format!r}")

    @classmethod
    def parse_cif(cls, lines: Iterable[str]) -> Self:
        if isinstance(lines, (str)):
            lines = [lines]
        block = unwrap(parse_cif("\n".join(lines)))

        logging.warning(
            "PDBx/mmCIF files include chemical information that Pablo ignores."
            + " For details, see 'How Pablo loads PDB files' in the docs.",
        )

        n_atom_sites = len(block["_atom_site.id"])

        data = cls(
            strict=True,
            cryst1_a=unwrap_or_none(cif_opt_floats(block["_cell.length_a"])),
            cryst1_b=unwrap_or_none(cif_opt_floats(block["_cell.length_b"])),
            cryst1_c=unwrap_or_none(cif_opt_floats(block["_cell.length_c"])),
            cryst1_alpha=unwrap_or_none(cif_opt_floats(block["_cell.angle_alpha"])),
            cryst1_beta=unwrap_or_none(cif_opt_floats(block["_cell.angle_beta"])),
            cryst1_gamma=unwrap_or_none(cif_opt_floats(block["_cell.angle_gamma"])),
            model=cif_opt_ints(block["_atom_site.pdbx_PDB_model_num"]),
            serial=cif_strs(block["_atom_site.id"]),
            name=cif_strs(block["_atom_site.label_atom_id"]),
            alt_loc=cif_strs(block["_atom_site.label_alt_id"]),
            chain_id=cif_strs(block["_atom_site.label_asym_id"]),
            res_seq=cif_strs(block["_atom_site.label_seq_id"]),
            i_code=cif_strs(block["_atom_site.pdbx_PDB_ins_code"]),
            x=cif_floats(block["_atom_site.Cartn_x"]),
            y=cif_floats(block["_atom_site.Cartn_y"]),
            z=cif_floats(block["_atom_site.Cartn_z"]),
            occupancy=cif_floats(block["_atom_site.occupancy"]),
            temp_factor=cif_floats(block["_atom_site.B_iso_or_equiv"]),
            element=cif_strs(block["_atom_site.type_symbol"]),
            charge=cif_opt_ints(block["_atom_site.pdbx_formal_charge"]),
            line_no=cif_ints(block["_atom_site.id.__pablo__line_no"]),
            # TODO: See if we can improve the following
            terminated=[False] * n_atom_sites,
            conects=[set()] * n_atom_sites,
        )
        # TODO: CONECT record equivalent

        residues = [
            (res_seq, res_name, chain_id)
            for res_seq, res_name, chain_id in zip(
                block["_pdbx_poly_seq_scheme.seq_id"],
                block["_pdbx_poly_seq_scheme.mon_id"],
                block["_pdbx_poly_seq_scheme.asym_id"],
            )
        ]
        model_residues = list(residues)
        res_seq, res_name, chain_id = model_residues.pop(0)
        prev_model: type[__UNSET__] | int | None = __UNSET__
        for i, (this_res_seq, this_chain_id, this_model) in enumerate(
            zip(data.res_seq, data.chain_id, data.model, strict=True),
        ):
            # Reset model_residues on new model
            if prev_model is not __UNSET__ and prev_model != this_model:
                model_residues = list(residues)
                res_seq, res_name, chain_id = model_residues.pop(0)
            # Pop the next residue off the stack when the residue changes
            if this_res_seq != res_seq or this_chain_id != chain_id:
                res_seq, res_name, chain_id = model_residues.pop(0)
            # Raise an error if we skip a residue
            if this_res_seq != res_seq or this_chain_id != chain_id:
                this_serial = data.serial[i]
                this_line_no = data.line_no[i]
                raise PabloError(
                    "Could not identify residue name"
                    + f" for atom serial {this_serial} (l{this_line_no})",
                )
            data.res_name.append(res_name)
            prev_model = this_model

        return data

    @classmethod
    def parse_pdb(cls, lines: Sequence[str], strict: bool = False) -> Self:
        """
        Parse PDB file lines into a new ``PdbData`` object.

        Parameters
        ----------
        lines
            An iterable of strings containing the lines from a PDB file
        strict
            Whether to enforce strict PDB format compliance
        """
        model_n = None
        data = cls(strict=strict)
        for i, line in enumerate(lines):
            data._cursor = i + 1
            if line.startswith("MODEL "):
                model_n = int(line[10:14])
            if line.startswith("ENDMDL "):
                model_n = None
            if line.startswith(("ATOM  ", "HETATM")):
                data._append_coord_line(line)
                data.model[-1] = model_n
            if line.startswith("TER   "):
                data.terminated[-1] = True
            if line.startswith("CRYST1"):
                data.cryst1_a = float(line[6:15])
                data.cryst1_b = float(line[15:24])
                data.cryst1_c = float(line[24:33])
                data.cryst1_alpha = float(line[33:40])
                data.cryst1_beta = float(line[40:47])
                data.cryst1_gamma = float(line[47:54])

        data._cursor = 0

        # Read all CONECT records
        data.conects = cls._process_conects(
            lines,
            data.serial_to_index,
            data.conects,
            data.model,
        )

        return data

    @property
    def n_atoms(self) -> int:
        """The number of ATOM/HETATM records in the first model"""
        try:
            first_model = self.model[0]
        except IndexError:
            return 0
        return sum(1 for model in self.model if model == first_model)

    def _append_coord_line(self, line: str):
        """
        Append data from an ATOM or HETATM line to the internal data structures.

        Parameters
        ----------
        line
            An ATOM or HETATM line from a PDB file
        """
        for field_ in dataclasses.fields(self):
            value = getattr(self, field_.name)
            if hasattr(value, "append"):
                value.append(__UNSET__)
                assert value[-1] is __UNSET__

        self.line_no[-1] = self._cursor
        self.model[-1] = None
        self.serial[-1] = line[6:11].strip()
        self.serial_to_index[self.serial[-1]].append(len(self.serial) - 1)
        self.name[-1] = line[12:16].strip()
        self.alt_loc[-1] = line[16].strip() or ""
        self.res_name[-1] = (
            line[17:20].strip()
            if self.strict or any(char.isspace() for char in line[17:21])
            else line[17:21]
        )
        self.chain_id[-1] = line[21].strip()
        self.res_seq[-1] = line[22:26].strip()
        self.i_code[-1] = line[26].strip() if line[26].strip() else " "
        self.x[-1] = float(line[30:38])
        self.y[-1] = float(line[38:46])
        self.z[-1] = float(line[46:54])
        self.occupancy[-1] = float(line[54:60])
        self.temp_factor[-1] = float(line[60:66])
        self.element[-1] = line[76:78].strip()
        self.charge[-1] = charge_int_or_none(line[78:80].strip(), strict=self.strict)
        self.terminated[-1] = False
        self.conects[-1] = set()

        # Ensure we've assigned a value to every field
        for field_ in dataclasses.fields(self):
            value = getattr(self, field_.name)
            if hasattr(value, "append"):
                assert value[-1] is not __UNSET__

    @staticmethod
    def _process_conects(
        lines: Iterable[str],
        serial_to_index: dict[str, list[int]],
        conects: list[set[int]],
        model: Sequence[int | None],
    ) -> list[set[int]]:
        """
        Process CONECT records from the lines of a PDB file.

        This method processes CONECT records while supporting multiple models in
        the PDB file. Each atom serial number used in CONECT records should
        appear no more than once per model, ensuring unambiguous connectivity.
        PDB files that violate this invariant (e.g., having the same serial used
        for multiple atoms within a single model) raise
        ``UnknownOrAmbiguousSerialInConectError``.

        Parameters
        ----------
        lines
            An iterable of strings containing the lines from a PDB file
        serial_to_index
            Mapping from atom serial numbers to their indices in the data
            structure
        conects
            List of sets representing connectivity for each atom. The ith set
            contains the atom indices connected to atom ``i``.
        model
            Sequence indicating which model each atom belongs to

        Returns
        -------
        list[set[int]]
            Updated connectivity information as a list of sets
        """
        for line in lines:
            if line.startswith("CONECT"):
                # a is the serial of the first atom in the conect, we need its indices
                a = line[6:11].strip()
                a_idcs = serial_to_index.get(a, [])

                # Conects are usually provided once for multi-model files
                # Each serial in CONECT records should appear exactly once per
                # model
                a_models = [model[i] for i in a_idcs]
                if len(set(a_models)) != len(a_models):
                    raise UnknownOrAmbiguousSerialInConectError(a, a_idcs)

                for a_idx, a_model in zip(a_idcs, a_models):
                    for start, stop in [(11, 16), (16, 21), (21, 26), (26, 31)]:
                        if len(line.strip()) <= start:
                            continue

                        b = line[start:stop].strip()

                        b_idcs = [
                            i for i in serial_to_index.get(b, []) if model[i] == a_model
                        ]
                        if len(b_idcs) != 1:
                            raise UnknownOrAmbiguousSerialInConectError(b, b_idcs)
                        b_idx = b_idcs[0]

                        conects[a_idx].add(b_idx)
                        conects[b_idx].add(a_idx)
        return conects

    @property
    def residue_indices(self) -> Iterator[tuple[int, ...]]:
        """
        Get an iterator over the indices of atoms belonging to each residue.

        Yields
        ------
        A tuple of atom indices representing a single residue
        """
        if len(self.model) == 0:
            return
        first_model: int | None = self.model[0]
        indices = []
        prev = None
        res_idx = 0
        self.res_idx = []
        for atom_idx, (alt_loc, terminated, *residue_info) in enumerate(
            zip(
                self.alt_loc,
                self.terminated,
                self.model,
                self.res_name,
                self.chain_id,
                self.res_seq,
                self.i_code,
                strict=True,
            ),
        ):
            if alt_loc != "":
                # TODO: Support alt-locs as alternate conformers
                warnings.warn(
                    "Alt locs not supported; only empty or 'A' alt locs will be read",
                )
                if alt_loc != "A":
                    # TODO: Improve this res_idx behavior
                    self.res_idx.append(res_idx)
                    continue
            if residue_info[0] != first_model:
                # TODO: Support multi-model files
                warnings.warn(
                    "Multi-model files not supported; topology will reflect first model",
                )
                break
            if prev == residue_info or prev is None or len(indices) == 0:
                indices.append(atom_idx)
            else:
                yield tuple(indices)
                res_idx += 1
                indices = [atom_idx]

            self.res_idx.append(res_idx)

            if terminated:
                yield tuple(indices)
                indices = []
                res_idx += 1

            prev = residue_info

        if len(indices) > 0:
            yield tuple(indices)

    def subset_matches_residue(
        self,
        res_atom_idcs: Collection[int],
        residue_definition: ResidueDefinition,
    ) -> PossibleResidueMatch:
        """
        Check if a set of atom indices matches a given residue definition.

        This performs a name-based match; connectivity-based matches occur
        later. Virtual sites are strictly enforced to match the residue
        definition exactly - missing or extra virtual sites will cause a
        mismatch.

        Parameters
        ----------
        res_atom_idcs
            Indices of atoms in the PDB file to check for matching
        residue_definition
            The residue definition to match against
        """
        # Raise an error if the match would be empty - this way the
        # return value's truthiness always reflects whether there was a match
        if len(res_atom_idcs) == 0:
            raise PabloError("cannot match empty res_atom_idcs")

        logging.debug(f"  Attempting match against {residue_definition.description}")

        # Drop virtual sites
        unmatched_vsites = list(residue_definition.virtual_sites)
        matched_vsite_indices: list[int] = []
        res_atom_idcs_without_vsites: list[int] = []
        for i in tuple(res_atom_idcs):
            name = self.name[i]
            if name in residue_definition.virtual_sites:
                try:
                    unmatched_vsites.pop(unmatched_vsites.index(name))
                except (ValueError, IndexError):
                    reason = f"Required virtual site {name} appeared too many times"
                    logging.debug("    Match failed: " + reason)
                    return ResidueMismatch(
                        residue_definition=residue_definition,
                        index_to_atomdef={j: None for j in res_atom_idcs},
                        reason=reason,
                    )
                else:
                    matched_vsite_indices.append(i)
            else:
                res_atom_idcs_without_vsites.append(i)
        if len(unmatched_vsites) != 0:
            reason = f"Required virtual sites not found in PDB file: {', '.join(unmatched_vsites)}"
            logging.debug("    Match failed: " + reason)
            return ResidueMismatch(
                residue_definition=residue_definition,
                index_to_atomdef={i: None for i in res_atom_idcs},
                reason=reason,
            )
        if len(matched_vsite_indices) != 0:
            logging.info(
                f"    Match to {residue_definition.description} dropped virtual sites on",
            )
            logging.info(
                f"    lines {', '.join(str(self.line_no[j]) for j in matched_vsite_indices)}",
            )
        res_atom_idcs = res_atom_idcs_without_vsites

        # Skip definitions with too few atoms
        if len(residue_definition.atoms) < len(res_atom_idcs):
            reason = f"Too few atoms in residue definition ({len(residue_definition.atoms)} < {len(res_atom_idcs)})"
            logging.debug("    Match failed: " + reason)
            return ResidueMismatch(
                residue_definition=residue_definition,
                index_to_atomdef={i: None for i in res_atom_idcs},
                reason=reason,
            )

        # Skip non-(cross)linking definitions with the wrong number of atoms
        if (
            residue_definition.linking_bond is None
            and residue_definition.crosslink is None
            and len(
                residue_definition.atoms,
            )
            != len(res_atom_idcs)
        ):
            reason = "No links and wrong number of atoms"
            logging.debug("    Match failed: " + reason)
            return ResidueMismatch(
                residue_definition=residue_definition,
                index_to_atomdef={i: None for i in res_atom_idcs},
                reason=reason,
            )

        # Get the map from the canonical names to the indices
        index_to_atomdef = {
            i: residue_definition.name_to_atom.get(self.name[i], None)
            for i in res_atom_idcs
        }
        if not no_none_in_values(index_to_atomdef):
            matched_atoms = {
                atom.name for atom in index_to_atomdef.values() if atom is not None
            }
            reason = (
                "The following atoms had unknown names: "
                + ", ".join(
                    self.name[i] for i, atom in index_to_atomdef.items() if atom is None
                )
                + f" (expected {'; '.join(sorted((atom.name + (' or synonyms ' if atom.synonyms else '') + ', '.join(atom.synonyms) for atom in residue_definition.atoms if atom.name not in matched_atoms), key=len))})"
            )
            logging.debug("    Match failed: " + reason)
            return ResidueMismatch(
                residue_definition=residue_definition,
                index_to_atomdef=index_to_atomdef,
                reason=reason,
            )

        match = ResidueMatch(
            index_to_atomdef=dict(index_to_atomdef),
            residue_definition=residue_definition,
            vsite_idcs=tuple(matched_vsite_indices),
        )

        matched_atoms = {atom.name for atom in index_to_atomdef.values()}

        # Fail to match if any atoms in PDB file got matched to more than one name
        if len(matched_atoms) != len(res_atom_idcs):
            reason = "Atom definition matched multiple PDB coordinate records"
            logging.debug("    Match failed: " + reason)
            return match.reject(reason)

        # This assert should be guaranteed by the above
        assert set(index_to_atomdef.keys()) == set(res_atom_idcs)

        # Check that elements match, but tolerate missing columns and wrong case
        if any(
            self.element[i] != "" and self.element[i].lower() != atom.symbol.lower()
            for i, atom in index_to_atomdef.items()
        ):
            reason = "Element mismatch"
            logging.debug("    Match failed: " + reason)
            return match.reject(reason)

        # PDB files can not be trusted; ignore charges.
        # # Check that charges match, but tolerate missing columns
        # if any(
        #     self.charge[i] is not None and self.charge[i] != atom.charge
        #     for i, atom in index_to_atomdef.items()
        # ):
        #     logging.debug("    Match failed: Charge mismatch")
        #     return None

        missing_atoms = [
            atom for atom in residue_definition.atoms if atom.name not in matched_atoms
        ]

        # Match only if all the leaving atoms associated with each linking atom
        # is either entirely present or entirely absent
        missing_atom_names = {atom.name for atom in missing_atoms}
        if any(not atom.leaving for atom in missing_atoms):
            reason = "Missing atom(s) are not leaving atoms"
            reason += (
                f" {tuple({atom.name for atom in missing_atoms if not atom.leaving})}"
            )
            logging.debug("    Match failed: " + reason)
            return match.reject(reason)
        elif residue_definition._missing_atoms_are_valid_leaving_atoms(
            missing_atom_names,
        ):
            logging.debug("    Match succeeded!")
            return match
        else:
            reason = "Missing atoms do not specify link"
            logging.debug("    Match failed: " + reason)
            return match.reject(reason)

    @cached_property
    def atom_idx_to_res_idx(self) -> dict[int, int]:
        """
        A mapping from atom indices to their corresponding residue index.
        """
        value: dict[int, int] = {}
        for res_idx, atom_indices in enumerate(self.residue_indices):
            for atom_idx in atom_indices:
                value[atom_idx] = res_idx
        return value

    def get_name_based_matches(
        self,
        residue_library: Mapping[str, Iterable[ResidueDefinition]],
    ) -> Iterator[list[PossibleResidueMatch]]:
        """
        Get possible matches for residues based on their names.

        A list of possible residue matches is yielded for each residue in the
        PDB file.

        Parameters
        ----------
        residue_library
            A mapping from residue names to their possible definitions
        """
        for res_atom_idcs in self.residue_indices:
            prototype_index = res_atom_idcs[0]
            res_name = self.res_name[prototype_index]
            logging.info(f"Beginning name-based match of {res_name} {res_atom_idcs}")
            if len(res_atom_idcs) <= 3:
                logging.debug(
                    f"  Atom names are ({', '.join(self.name[i] for i in res_atom_idcs)})",
                )

            matches = [
                self.subset_matches_residue(
                    res_atom_idcs,
                    residue_definition,
                )
                for residue_definition in residue_library.get(res_name, [])
            ]
            if len(matches) > 0:
                yield matches
            else:
                yield [
                    NoResidueDefinitions(
                        residue_definition=res_name,
                        index_to_atomdef={i: None for i in res_atom_idcs},
                    ),
                ]

    @staticmethod
    def _get_prev_next(
        i: int,
        all_matches: Sequence[Sequence[PossibleResidueMatch]],
    ) -> tuple[Sequence[PossibleResidueMatch], Sequence[PossibleResidueMatch]]:
        """
        Get the previous and next matches for a given residue index.

        Parameters
        ----------
        i
            The current residue index
        all_matches
            All possible matches for each residue

        Returns
        -------
        (this_matches, next_matches)
            A tuple containing the previous and next matches
        """
        prev_matches: Sequence[PossibleResidueMatch] = (
            () if i == 0 else all_matches[i - 1]
        )
        next_matches: Sequence[PossibleResidueMatch] = (
            () if len(all_matches) <= i + 1 else all_matches[i + 1]
        )
        return prev_matches, next_matches

    def rescue_partial_matches_with_conect_records(
        self,
        this_res_idx: int,
        all_matches: Sequence[Sequence[PossibleResidueMatch]],
    ) -> Iterator[PossibleResidueMatch]:
        """
        Attempt to rescue partial matches using CONECT records.

        Parameters
        ----------
        this_res_idx
            The index of the current residue
        all_matches
            All possible matches for each residue
        """
        this_matches = all_matches[this_res_idx]
        prev_matches, next_matches = self._get_prev_next(this_res_idx, all_matches)

        # Don't try to rescue if there is a successful name-based match
        if any(isinstance(match, SuccessfulMatch) for match in this_matches):
            yield from this_matches
            return

        logging.info(
            f"Beginning link-based match of {self.res_name[this_matches[0].prototype_index]} {this_matches[0].res_atom_idcs}",
        )

        res_atom_idcs = this_matches[0].res_atom_idcs

        # in conects, node data is the pdb index, edge data is the sorted
        # pdb indices of the two atoms in the CONECT record
        # bonds to other residues are ignored - we'll deal with them later
        conects = Graph[int, tuple[int, int]]()
        conects.add_nodes_from(res_atom_idcs)
        external_conects: list[tuple[int, int]] = []
        for pdb_idx in res_atom_idcs:
            for j in self.conects[pdb_idx]:
                if j not in res_atom_idcs:
                    external_conects.append((pdb_idx, j))
                    continue
                conects.add_edge(
                    pdb_idx,
                    j,
                    sort_tuple((pdb_idx, j)),
                )

        for match in this_matches:
            # Always yield the original mismatch as a record that this rescue
            # took place
            yield match
            if not isinstance(match, ResidueMismatch):
                # Only attempt to rescue ResidueMismatch
                continue

            resdef = match.residue_definition
            logging.debug(f"  Attempting to rescue {resdef.description}")
            if len(resdef.virtual_sites) != 0:
                # No CONECT-based matches for resdefs with virtual sites
                # TODO: Implement this?
                logging.debug("    REJECTED: Virtual sites not supported")
                continue

            # Check that there are roughly the right number of atoms
            if (
                len(res_atom_idcs) < resdef.n_core_atoms
                or len(res_atom_idcs) > resdef.n_expected_atoms
            ):
                logging.debug(
                    "    REJECTED: Wrong number of atoms "
                    + f"({len(res_atom_idcs)} should be in "
                    + f"[{resdef.n_core_atoms}, {resdef.n_expected_atoms}])",
                )
                continue

            logging.debug("    Compiling bonds graph")
            # Build up a graph of the internal bonds we expect from the residue definition
            bonds = Graph[AtomDefinition, BondDefinition]()
            bonds.add_nodes_from(resdef.atoms)
            bonds.add_edges_from(
                (resdef.name_to_atom[bond.atom1], resdef.name_to_atom[bond.atom2], bond)
                for bond in resdef.bonds
            )

            logging.debug(f"    {bonds.n_nodes=} {conects.n_nodes=}")
            if bonds.is_subgraph_of(conects):
                logging.debug("    subgraph exists without node matcher")
            else:
                logging.debug("    subgraph does not exist without node matcher")

            # Iterate over subgraphs of bonds that match all of conects
            logging.debug(
                "    Find matching subgraphs",
            )
            rescued: bool = False
            for mapping_idx, mapping in enumerate(
                bonds.get_mappings(
                    conects,
                    node_matcher=lambda b, c: (
                        self.element[c] == b.symbol
                        and (
                            (
                                self.name[c] in (b.name, *b.synonyms)
                                and not resdef.is_anonymous
                            )
                            or (self.charge[c] is None and not self.strict)
                            or ((self.charge[c] or 0) == b.charge)
                        )
                    ),
                    subgraph=True,  # Look for a subgraph of bonds isomorphic to conects
                    induced=True,  # CONECT records must all be present to find a match
                ),
            ):
                mapping = {v: k for k, v in mapping.items()}
                logging.debug(
                    f"      mapping_{mapping_idx}={ {i: atom.name for i, atom in mapping.items()}!r}",
                )
                assert set(mapping) == set(match.res_atom_idcs)

                # Check that all missing atoms are leaving atoms
                # This is done before missing_atoms is created so that any can
                # short circuit (5x speedup worst case)
                if any(
                    not atom.leaving
                    for atom in resdef.atoms
                    if atom not in mapping.values()
                ):
                    logging.debug(
                        "        REJECTED: Missing atoms are not leaving atoms",
                    )
                    continue

                missing_atoms = {
                    atom for atom in resdef.atoms if atom not in mapping.values()
                }
                logging.debug(f"      {missing_atoms=}")

                if (
                    # Missing atoms must exactly define linking bonds
                    not resdef._missing_atoms_are_valid_leaving_atoms(
                        {atom.name for atom in missing_atoms},
                    )
                ):
                    logging.debug(
                        "        REJECTED: Missing atoms are not a valid set of leaving atoms",
                    )
                    continue

                match = ResidueConectMatch(
                    residue_definition=resdef,
                    index_to_atomdef=mapping,
                    vsite_idcs=(),
                )

                # Check that the CONECTs between this residue and another are
                # compatible with this mapping
                prior_conects: list[tuple[int, int]] = []
                posterior_conects: list[tuple[int, int]] = []
                crosslink_conects: list[tuple[int, int]] = []
                unknown_conects: list[tuple[int, int]] = []
                for pdb_idx, j in external_conects:
                    j_name = self.name[j]
                    i_name = mapping[pdb_idx].name
                    logging.debug(
                        f"        {pdb_idx, j}: {(i_name, j_name)} {len(prev_matches)=} {len(next_matches)=}",
                    )
                    logging.debug(f"          {resdef.possible_prior_bond_names=}")
                    logging.debug(f"          {resdef.possible_posterior_bond_names=}")
                    logging.debug(f"          {resdef.possible_crosslink_bond_names=}")
                    if (
                        len(prev_matches) > 0
                        and j in prev_matches[0].res_atom_idcs
                        and (j_name, i_name) in resdef.possible_prior_bond_names
                    ):
                        logging.debug("          prior")
                        prior_conects.append((j, pdb_idx))
                    elif (
                        len(next_matches) > 0
                        and j in next_matches[0].res_atom_idcs
                        and (i_name, j_name) in resdef.possible_posterior_bond_names
                    ):
                        logging.debug("          posterior")
                        posterior_conects.append((pdb_idx, j))
                    elif (i_name, j_name) in resdef.possible_crosslink_bond_names:
                        logging.debug("          crosslink")
                        crosslink_conects.append((pdb_idx, j))
                    else:
                        logging.debug("          unknown")
                        unknown_conects.append((pdb_idx, j))
                logging.debug(f"      {unknown_conects=}")
                logging.debug(f"      {posterior_conects=}")
                logging.debug(f"      {prior_conects=}")
                logging.debug(f"      {crosslink_conects=}")

                if (  # nofmt
                    # If there are conect records between this residue and another, match must agree
                    len(unknown_conects) > 0
                    or len(posterior_conects) > 1
                    or len(prior_conects) > 1
                    or len(crosslink_conects) > 1
                    or (bool(crosslink_conects) and not match.expects_crosslink)
                    or (bool(posterior_conects) and not match.expects_posterior_bond)
                    or (bool(prior_conects) and not match.expects_prior_bond)
                ):
                    logging.debug(
                        "        REJECTED: Interresidue CONECT records not compatible with missing leaving atoms",
                    )
                    continue

                if bool(crosslink_conects):
                    match.set_crosslink(*unwrap(crosslink_conects))
                if bool(posterior_conects):
                    match.set_posterior_bond(*unwrap(posterior_conects))
                if bool(prior_conects):
                    match.set_prior_bond(*unwrap(prior_conects))
                # Yield multiple residue matches for multiple mappings and let
                # the remaining filters sort it out; if we end up with multiple
                # matches at the end of filtering, then we couldn't
                # unambiguously rescue this match
                logging.debug("        ACCEPTED")
                rescued = True
                yield match

            if not rescued:
                logging.debug("    REJECTED: no mappings matched")

    def identify_polymer_linkages(
        self,
        this_res_idx: int,
        all_matches: Sequence[Sequence[PossibleResidueMatch]],
    ) -> Iterator[PossibleResidueMatch]:
        """
        Identify linkages between neighbouring residues of a polymer chain.

        This method determines whether adjacent residues in a chain can be
        linked by examining the connectivity, implied by which leaving atoms are
        missing from the match, of the this and the neighbouring residues'
        matches. ``TER`` records are treated as strict boundaries - no polymer
        linkage (prior or posterior bond) can form across them.

        Parameters
        ----------
        this_res_idx
            The index of the current residue
        all_matches
            All possible matches for each residue
        """
        this_matches = all_matches[this_res_idx]
        prev_matches, next_matches = self._get_prev_next(this_res_idx, all_matches)

        if len(list(only_matched(this_matches))) != 0:
            logging.info(
                f"Beginning link-based match of {self.res_name[this_matches[0].prototype_index]} {this_matches[0].res_atom_idcs}",
            )
        else:
            yield from this_matches
            return
        # neighbour_supported_posterior_bonds and
        # neighbour_supported_prior_bonds are maps from possible linking
        # bonds in this residue to the atom index (in the neighbouring
        # residue) where the linking atom can be found
        neighbour_supported_posterior_bonds: dict[BondDefinition, int] = {
            next_match.residue_definition.linking_bond: next_match.canonical_atom_name_to_index[
                next_match.residue_definition.prior_bond_linking_atom
            ]
            for next_match in only_matched(next_matches)
            if isinstance(next_match, ResidueMatch)
            and next_match.residue_definition.linking_bond is not None
            and next_match.expects_prior_bond
        }
        neighbour_supported_prior_bonds: dict[BondDefinition, int] = {
            prev_match.residue_definition.linking_bond: prev_match.canonical_atom_name_to_index[
                prev_match.residue_definition.posterior_bond_linking_atom
            ]
            for prev_match in only_matched(prev_matches)
            if isinstance(prev_match, ResidueMatch)
            and prev_match.residue_definition.linking_bond is not None
            and prev_match.expects_posterior_bond
        }

        valid_next_matches = list(only_matched(next_matches))
        neighbours_support_molecule_end = (
            any(not next_match.expects_prior_bond for next_match in valid_next_matches)
            or len(valid_next_matches) == 0
        )
        valid_prev_matches = list(only_matched(prev_matches))
        neighbours_support_molecule_start = (
            any(
                not prev_match.expects_posterior_bond
                for prev_match in valid_prev_matches
            )
            or len(valid_prev_matches) == 0
        )
        prev_residue_terminated = (
            len(prev_matches) > 0 and self.terminated[prev_matches[0].prototype_index]
        )
        this_residue_terminated = self.terminated[this_matches[0].prototype_index]
        if len(list(only_matched(this_matches))) != 0:
            logging.debug(
                f"  {neighbour_supported_posterior_bonds=}",
            )
            logging.debug(
                f"  {neighbour_supported_prior_bonds=}",
            )
            logging.debug(
                f"  {neighbours_support_molecule_end=}",
            )
            logging.debug(
                f"  {neighbours_support_molecule_start=}",
            )
        for match in this_matches:
            if not isinstance(match, ResidueMatch):
                yield match
                continue
            logging.debug(
                f"  Attempting link-based match against {match.residue_definition.description}",
            )
            if match.expects_prior_bond:
                assert match.residue_definition.linking_bond is not None
                prior_partner_name = match.residue_definition.linking_bond.atom1
                prior_linking_name = match.residue_definition.linking_bond.atom2
                prior_bond_linking_atom_idx = match.canonical_atom_name_to_index[
                    prior_linking_name
                ]
                prior_conect_partners = [
                    (self.name[i], i)
                    for i in self.conects[prior_bond_linking_atom_idx]
                    if self.name[i] == prior_partner_name
                    and i not in match.res_atom_idcs
                ]
                if len(prior_conect_partners) > 1:
                    i = match.prototype_index
                    res = f"{self.chain_id[i]}:{self.res_name[i]}{self.res_seq[i]}"
                    line = self.line_no[i]
                    logging.info(
                        "    Multiple possible prior linking bond partners"
                        + f" found in CONECT records for residue {res} ({line});"
                        + " ignoring all and proceeding with standard linkage"
                        + "analysis",
                    )

                if len(prior_conect_partners) == 1:
                    logging.debug("    Prior linking bond found in CONECT records")
                    ((_, prior_bond_partner_atom_idx),) = prior_conect_partners
                    match.set_prior_bond(
                        prior_bond_partner_atom_idx,
                        prior_bond_linking_atom_idx,
                    )
                elif (
                    match.residue_definition.linking_bond
                    not in neighbour_supported_prior_bonds
                ):
                    reason = "Prior bond expected but not supported by neighbours"
                    logging.debug(f"    Match failed: {reason}")
                    yield match.reject(reason)
                    continue
                elif prev_residue_terminated:
                    reason = "Prior bond expected but cannot form polymer bond across TER record"
                    logging.debug(f"    Match failed: {reason}")
                    yield match.reject(reason)
                    continue
                else:
                    bond_idcs = (
                        neighbour_supported_prior_bonds[
                            match.residue_definition.linking_bond
                        ],
                        prior_bond_linking_atom_idx,
                    )
                    if (
                        match.prior_bond_idcs is not None
                        and match.prior_bond_idcs != bond_idcs
                    ):
                        reason = "Multiple inconsistent prior bonds identified"
                        logging.debug(f"    Match failed: {reason}")
                        yield match.reject(reason)
                        continue
                    match.set_prior_bond(*bond_idcs)
            elif not neighbours_support_molecule_start:
                reason = "Prior bond not expected but required by neighbours"
                logging.debug(f"    Match failed: {reason}")
                yield match.reject(reason)
                continue

            if match.expects_posterior_bond:
                assert match.residue_definition.linking_bond is not None
                posterior_linking_name = match.residue_definition.linking_bond.atom1
                posterior_partner_name = match.residue_definition.linking_bond.atom2
                posterior_bond_linking_atom_idx = match.canonical_atom_name_to_index[
                    posterior_linking_name
                ]
                posterior_conect_partners = [
                    (self.name[i], i)
                    for i in self.conects[posterior_bond_linking_atom_idx]
                    if self.name[i] == posterior_partner_name
                    and i not in match.res_atom_idcs
                ]
                if len(posterior_conect_partners) > 1:
                    i = match.prototype_index
                    res = f"{self.chain_id[i]}:{self.res_name[i]}{self.res_seq[i]}"
                    line = self.line_no[i]
                    logging.info(
                        "    Multiple possible posterior linking bond partners"
                        + f" found in CONECT records for residue {res} ({line});"
                        + " ignoring all and proceeding with standard linkage"
                        + " analysis",
                    )

                if len(posterior_conect_partners) == 1:
                    logging.debug(
                        "    Unique posterior linking bond found in CONECT records",
                    )
                    ((_, posterior_bond_partner_atom_idx),) = posterior_conect_partners
                    match.set_posterior_bond(
                        posterior_bond_linking_atom_idx,
                        posterior_bond_partner_atom_idx,
                    )
                elif (
                    match.residue_definition.linking_bond
                    not in neighbour_supported_posterior_bonds
                ):
                    reason = "Posterior bond expected but not supported by neighbours"
                    logging.debug(f"    Match failed: {reason}")
                    yield match.reject(reason)
                    continue
                elif this_residue_terminated:
                    reason = "Posterior bond expected but cannot form polymer bond across TER record"
                    logging.debug(f"    Match failed: {reason}")
                    yield match.reject(reason)
                    continue
                else:
                    bond_idcs = (
                        match.canonical_atom_name_to_index[
                            match.residue_definition.posterior_bond_linking_atom
                        ],
                        neighbour_supported_posterior_bonds[
                            match.residue_definition.linking_bond
                        ],
                    )
                    if (
                        match.posterior_bond_idcs is not None
                        and match.posterior_bond_idcs != bond_idcs
                    ):
                        reason = "Multiple inconsistent posterior bonds identified"
                        logging.debug(f"    Match failed: {reason}")
                        yield match.reject(reason)
                        continue
                    match.set_posterior_bond(*bond_idcs)
            elif not neighbours_support_molecule_end:
                reason = "Posterior bond not expected but required by neighbours"
                logging.debug(f"    Match failed: {reason}")
                yield match.reject(reason)
                continue

            logging.debug("    Accepted")
            yield match

    def filter_on_crosslinks(
        self,
        this_res_idx: int,
        all_matches: Sequence[Sequence[PossibleResidueMatch]],
    ) -> Iterator[PossibleResidueMatch]:
        """
        Identify crosslinks between residues and filter out broken crosslinks.

        Parameters
        ----------
        this_res_idx : int
            The index of the current residue
        all_matches : Sequence[Sequence[PossibleResidueMatch]]
            All possible matches for each residue
        """
        this_matches = all_matches[this_res_idx]

        # Check for crosslinks
        # TODO: This could be simplified if we required crosslinking atoms not to have synonyms
        if len(list(only_matched(this_matches))) != 0:
            logging.info(
                f"Assigning crosslinks for {self.res_name[this_matches[0].prototype_index]} {this_matches[0].res_atom_idcs}",
            )
        else:
            yield from this_matches
            return
        for match in this_matches:
            if not isinstance(match, ResidueMatch):
                yield match
                continue
            logging.debug(
                f"  Attempting crosslink-based match against {match.residue_definition.description}",
            )
            if match.crosslink_idcs is not None:
                # This match's crosslink has already been assigned
                logging.debug(
                    "    Skipping (crosslink already assigned)",
                )
                yield match
                continue
            if not match.expects_crosslink:
                logging.debug(
                    "    Skipping (crosslink not expected)",
                )
                yield match
                continue
            this_crosslink_def = match.residue_definition.crosslink
            if this_crosslink_def is None:
                # No crosslink defined for this match
                logging.debug(
                    "    Skipping (crosslink not expected 2: electric boogaloo)",
                )
                yield match
                continue
            this_crosslink_atom_idx = match.canonical_atom_name_to_index[
                this_crosslink_def.atom1
            ]

            this_crosslink_conects = self.conects[this_crosslink_atom_idx]
            for other_crosslink_atom_idx in this_crosslink_conects:
                logging.debug(
                    f"    checking possible partner {other_crosslink_atom_idx}",
                )
                other_crosslink_res_idx = self.atom_idx_to_res_idx[
                    other_crosslink_atom_idx
                ]
                other_matches = list(all_matches[other_crosslink_res_idx])
                for other_match in other_matches:
                    if not isinstance(other_match, ResidueMatch):
                        continue
                    logging.debug(
                        f"      in {other_match.residue_definition.description}",
                    )
                    other_crosslink_def = other_match.residue_definition.crosslink
                    other_crosslink_atom_canonical_name = other_match.atom(
                        other_crosslink_atom_idx,
                    ).name
                    if (
                        other_match.expects_crosslink
                        and other_crosslink_def is not None
                        and other_crosslink_def.flipped() == this_crosslink_def
                        and other_crosslink_def.atom1
                        == other_crosslink_atom_canonical_name
                    ):
                        logging.debug(
                            "        crosslink found!",
                        )
                        # We've found a crosslink!
                        # TODO: What if there are multiple possible crosslinks?
                        #       ATM the last one is assigned, then rejected
                        #       because the other CONECT records are not
                        #       satisfied
                        match.set_crosslink(
                            this_crosslink_atom_idx,
                            other_crosslink_atom_idx,
                        )
                        other_match.set_crosslink(
                            other_crosslink_atom_idx,
                            this_crosslink_atom_idx,
                        )
            if match.expects_crosslink and match.crosslink_idcs is None:
                yield match.reject(
                    "crosslink expected but no matching crosslink partner could be found",
                )
            else:
                yield match

    def match_additional_definitions(
        self,
        this_res_idx: int,
        all_matches: Sequence[Sequence[PossibleResidueMatch]],
        additional_definitions: Iterable[ResidueDefinition],
    ) -> Iterator[PossibleResidueMatch]:
        """
        Attempt to match additional definitions against unmatched residues.

        Parameters
        ----------
        this_res_idx
            The index of the current residue
        all_matches
            All possible matches for each residue
        additional_definitions
            Additional residue definitions to match against
        """
        this_matches = all_matches[this_res_idx]

        yield from this_matches
        if not any(this_matches):
            logging.info(
                f"Matching additional_substructures to {self.res_name[this_matches[0].prototype_index]} {this_matches[0].res_atom_idcs}",
            )
            res_atom_idcs = this_matches[0].res_atom_idcs
            for residue_definition in additional_definitions:
                yield self.subset_matches_residue(
                    res_atom_idcs,
                    residue_definition,
                )
            logging.debug(
                "  Done",
            )

    def filter_on_conect_records(
        self,
        this_res_idx: int,
        all_matches: Sequence[Sequence[PossibleResidueMatch]],
    ) -> Iterator[PossibleResidueMatch]:
        """
        Filter out matches that are inconsistent with existing CONECT records.

        Parameters
        ----------
        this_res_idx
            The index of the current residue
        all_matches
            All possible matches for each residue
        """
        this_matches = all_matches[this_res_idx]

        logging.info(
            f"Filtering matches on CONECT records for {self.res_name[this_matches[0].prototype_index]} {this_matches[0].res_atom_idcs}",
        )
        for match in this_matches:
            if not isinstance(match, ResidueMatch):
                yield match
                continue

            logging.debug(
                f"  Checking match {match.description}",
            )

            expected_bonds: set[tuple[int, int]] = set()
            for bond in match.residue_definition.bonds:
                try:
                    atom1_idx = match.canonical_atom_name_to_index[bond.atom1]
                    atom2_idx = match.canonical_atom_name_to_index[bond.atom2]
                except KeyError:
                    # Bond is for a missing leaving atom
                    continue
                expected_bonds.add(sort_tuple((atom1_idx, atom2_idx)))
            if match.crosslink_idcs is not None:
                expected_bonds.add(sort_tuple(match.crosslink_idcs))
            if match.prior_bond_idcs is not None:
                expected_bonds.add(sort_tuple(match.prior_bond_idcs))
            if match.posterior_bond_idcs is not None:
                expected_bonds.add(sort_tuple(match.posterior_bond_idcs))

            found_conects = set(
                flatten(
                    (sort_tuple((i, j)) for j in self.conects[i])
                    for i in match.res_atom_idcs
                ),
            )
            if not found_conects.issubset(expected_bonds):
                logging.debug(
                    "    REJECTED: CONECT record but no bond",
                )
                yield match.reject(
                    "found CONECT record that could not be matched with a bond",
                )
            else:
                logging.debug(
                    "    ACCEPTED",
                )
                yield match

    def choose_polymer_bonds(
        self,
        this_res_idx: int,
        all_matches: Sequence[Sequence[PossibleResidueMatch]],
    ) -> Iterator[PossibleResidueMatch]:
        """
        Choose whether to form bonds between residues where it is still ambiguous.

        If residues are "adjacent" and capable of forming links, matches that
        don't link them are rejected; if they're not "adjacent", those that do
        attempt to form links are rejected. Residues ``a`` and ``b`` are
        considered adjacent if ``b``'s ``ATOM``/``HETATM`` records immediately
        follow ``a``'s, there is no ``TER`` record between ``a`` and ``b``, and
        ``a`` and ``b`` have the same chain identifier.

        Parameters
        ----------
        this_res_idx
            The index of the current residue
        all_matches
            All possible matches for each residue
        """
        this_matches = all_matches[this_res_idx]
        prev_matches, next_matches = self._get_prev_next(this_res_idx, all_matches)

        # TODO: Nail down the definition of "Adjacent"
        logging.info(
            f"Filtering matches for polymer linkages in {self.res_name[this_matches[0].prototype_index]} {this_matches[0].res_atom_idcs}",
        )

        this_ptype_idx = next(iter(this_matches)).prototype_index
        prev_ptype_idx = (
            prev_matches[0].prototype_index if len(prev_matches) > 0 else None
        )
        next_ptype_idx = (
            next_matches[0].prototype_index if len(next_matches) > 0 else None
        )

        this_chain = self.chain_id[this_ptype_idx]
        prev_chain = (
            self.chain_id[prev_ptype_idx] if prev_ptype_idx is not None else None
        )
        next_chain = (
            self.chain_id[next_ptype_idx] if next_ptype_idx is not None else None
        )

        this_terminated = self.terminated[this_ptype_idx]
        prev_terminated = (
            self.terminated[prev_ptype_idx] if prev_ptype_idx is not None else True
        )

        can_form_prior_bond = any(
            m.prior_bond_idcs is not None
            for m in only_matched(this_matches)
            if isinstance(m, ResidueMatch)
        )
        can_form_posterior_bond = any(
            m.posterior_bond_idcs is not None
            for m in only_matched(this_matches)
            if isinstance(m, ResidueMatch)
        )

        prev_is_adjacent = (
            prev_ptype_idx is not None
            and prev_chain is not None
            and prev_chain == this_chain
            and not prev_terminated
        )
        next_is_adjacent = (
            next_ptype_idx is not None
            and next_chain is not None
            and next_chain == this_chain
            and not this_terminated
        )

        logging.debug(
            f"  {can_form_prior_bond=}",
        )
        logging.debug(
            f"  {can_form_posterior_bond=}",
        )
        logging.debug(
            f"  {prev_is_adjacent=}",
        )
        logging.debug(
            f"  {next_is_adjacent=}",
        )

        for match in this_matches:
            if not isinstance(match, ResidueMatch):
                yield match
                continue

            logging.debug(
                f"  Checking {match.description}",
            )

            current_match_forms_prior_bond = match.prior_bond_idcs is not None
            current_match_forms_posterior_bond = match.posterior_bond_idcs is not None

            if (
                can_form_prior_bond
                and prev_is_adjacent
                and not current_match_forms_prior_bond
            ):
                reason = "adjacent residues in unterminated chain can be linked"
                logging.debug(
                    f"    REJECTED: {reason}",
                )
                yield match.reject(reason)
                continue
            elif (
                can_form_prior_bond
                and not prev_is_adjacent
                and current_match_forms_prior_bond
            ):
                reason = "non-adjacent residues in unterminated chain cannot be linked"
                logging.debug(
                    f"    REJECTED: {reason}",
                )
                yield match.reject(reason)
                continue
            if (
                can_form_posterior_bond
                and next_is_adjacent
                and not current_match_forms_posterior_bond
            ):
                reason = "adjacent residues in unterminated chain can be linked"
                logging.debug(
                    f"    REJECTED: {reason}",
                )
                yield match.reject(reason)
                continue
            elif (
                can_form_posterior_bond
                and not next_is_adjacent
                and current_match_forms_posterior_bond
            ):
                reason = "non-adjacent residues in unterminated chain cannot be linked"
                logging.debug(
                    f"    REJECTED: {reason}",
                )
                yield match.reject(reason)
                continue
            if match:
                logging.debug(
                    "    ACCEPTED",
                )
            yield match

    def match_residues(
        self,
        residue_library: Mapping[str, Iterable[ResidueDefinition]],
        additional_definitions: Iterable[ResidueDefinition],
    ) -> list[list[PossibleResidueMatch]]:
        """
        Match residues in the PDB file against a database of known residues.

        Returns all residue matches, failed or successful, for each residue. The
        outer returned list has one element for each residue in the PDB file.
        The inner lists may have variable lengths based on the number of matches
        that were attempted against that residue, but each element in any inner
        list matches the same ``ATOM``/``HETATM`` records.

        This method performs a multi-stage process to match residues:

        1. Identify residues in the PDB file by grouping atoms based on their
           residue name, residue sequence number, chain ID, and insertion code.
           TER records (which terminate both residues and chains) act as strict boundaries.
        2. Perform initial atom name-based matching against the residue database.
           Virtual sites are strictly enforced to match exactly - missing or extra
           virtual sites will cause a mismatch.
        3. Use CONECT records for connectivity, element, and formal charge-based matching
           for residues that found no name-based matches; ambiguous cases raise errors.
        4. Identify and filter by polymer linkages (for peptides, nucleic acids, etc.).
           TER records prevent polymer linkages from forming across them.
        5. Perform crosslink identification and filtering for residues like cysteine
           that can form disulfide bonds.
        6. Match against ``additional_substructures`` for any residues with no matches
           in the residue database.
        7. Filter by CONECT records to ensure internal connectivity matches expectations;
           missing CONECT records are accepted where possible.
        8. Choose appropriate polymer bonds between adjacent residues when it is still ambiguous;
           TER records prevent linkages from forming across them.
        9. Attempt to match unknown molecules against unmatched residues.

        Parameters
        ----------
        residue_library
            A mapping from residue names to their possible definitions
        additional_definitions
            Additional substructure definitions to match against the graph of
            unmatched atoms.

        Returns
        -------
        list[list[PossibleResidueMatch]]
            A nested list structure containing the matching results. The outer list has one
            element for each residue in the PDB file, while each inner list contains all
            possible matches (both successful and unsuccessful) for that residue.
        """
        matches = list(self.get_name_based_matches(residue_library))

        class Filter(Protocol):
            def __call__(
                self,
                this_res_idx: int,
                all_matches: Sequence[Sequence[PossibleResidueMatch]],
            ) -> Iterator[PossibleResidueMatch]: ...

        match_filters: list[Filter] = [
            # functools.partial(
            #     self.match_additional_definitions,
            #     additional_definitions=additional_definitions,
            # ),
            self.rescue_partial_matches_with_conect_records,
            self.identify_polymer_linkages,
            self.filter_on_crosslinks,
            self.filter_on_conect_records,
            self.choose_polymer_bonds,
        ]
        for match_filter in match_filters:
            matches = [
                list(
                    match_filter(
                        this_res_idx=i,
                        all_matches=matches,
                    ),
                )
                for i, _ in enumerate(matches)
            ]
        logging.info("------")
        return matches

    def get_successful_matches(
        self,
        residue_library: Mapping[str, Iterable[ResidueDefinition]],
        additional_definitions: Sequence[ResidueDefinition],
    ) -> list[SuccessfulMatch]:
        """
        Get one successful match for each residue in the PDB file, or raise an error.

        Parameters
        ----------
        residue_library
            A mapping from residue names to their possible definitions
        additional_definitions
            Additional substructure definitions to match against the graph of
            unmatched atoms.
        """
        logging.info(
            "Getting all residue matches",
        )

        # List of residue matches, one per residue. This list has one residue
        # match for each residue in the PDB file iff every residue could be
        # identified
        residues: list[SuccessfulMatch] = []
        errors: list[list[MismatchProtocol] | list[SuccessfulMatch]] = []
        all_residues_successful = True
        check_additional_definitions: None | bool = None
        unmatched_atoms: set[int] = set()
        for possible_residue_matches in self.match_residues(
            residue_library,
            additional_definitions,
        ):
            logging.debug(
                f"  Checking errors for {self.res_name[possible_residue_matches[0].prototype_index]} {possible_residue_matches[0].res_atom_idcs}",
            )

            matches: list[SuccessfulMatch] = []
            mismatches: list[MismatchProtocol] = []
            for possible_match in possible_residue_matches:
                if isinstance(possible_match, SuccessfulMatch):
                    matches.append(possible_match)
                else:
                    mismatches.append(possible_match)

            if len(matches) == 0:
                logging.debug(
                    "    POSSIBLE FAILURE: no successful matches",
                )
                # No matches, PDB loading has failed
                all_residues_successful = False
                unmatched_atoms.update(mismatches[0].res_atom_idcs)
                if check_additional_definitions is None:
                    check_additional_definitions = True
                errors.append(mismatches)
            elif len(matches) == 1:
                logging.debug(
                    "    SUCCESS: 1 successful match",
                )
                # 1 match, this residue has succeeded
                residues.append(unwrap(matches))
            elif all(a.agrees_with(b) for a, b in itertools.pairwise(matches)):
                logging.debug(
                    "    SUCCESS: multiple matches with consistent chemistry",
                )
                # Multiple matches that specify identical chemistry, this residue has succeeded
                residues.append(next(iter(matches)))
            else:
                logging.debug(
                    "    FAILURE: multiple matches with inconsistent chemistry; PDB loading has failed",
                )
                # Multiple matches that specify different chemistry, PDB loading has failed
                all_residues_successful = False
                check_additional_definitions = False
                unmatched_atoms.update(mismatches[0].res_atom_idcs)
                errors.append(matches)

        if all_residues_successful:
            logging.debug(
                "All residues successfully matched",
            )
            return residues

        if check_additional_definitions:
            logging.debug(f"Checking additional definitions {additional_definitions}")
            additional_matches = apply_additional_definitions(
                self,
                residues,
                additional_definitions,
            )
            unmatched_atoms -= {
                i
                for match in additional_matches
                for i in match.res_atom_idcs
                if match.index_to_atomdef[i].symbol != ""
            }
            if len(unmatched_atoms) == 0:
                return residues + additional_matches
            else:
                raise create_pdb_residue_match_error(
                    data=self,
                    errors=errors,
                    additional_definitions=additional_definitions,
                    additional_matches=additional_matches,
                    unmatched_pdb_idcs=unmatched_atoms,
                )

        raise create_pdb_residue_match_error(
            data=self,
            errors=errors,
            additional_definitions=additional_definitions,
        )

    def matches_to_rdmol(
        self,
        matches: Iterable[SuccessfulMatch],
        *,
        use_canonical_names: bool = False,
    ) -> RdMol:
        rdmol = EditableRdMol()
        pdb_idcs = {i for i in range(self.n_atoms) if self.alt_loc[i] in {"", " ", "A"}}
        bonds: dict[tuple[int, int], BondDefinition] = {}
        """{(pdb_idx_1, pdb_idx_2): bond_order, ...}"""
        atoms: dict[int, tuple[AtomDefinition, SuccessfulMatch]] = {}
        """{pdb_idx: (atom_definition, match), ...}"""

        logging.debug("Begin collating chemical information from matches")
        for match in matches:
            logging.debug(
                f"  collating {match.residue_definition.description} with {match.canonical_atom_name_to_index}",
            )
            for vsite in match.vsite_idcs:
                pdb_idcs.remove(vsite)

            for i, atom in match.index_to_atomdef.items():
                if atom.symbol == "" or i in atoms:
                    continue
                atoms[i] = (atom, match)
                pdb_idcs.remove(i)

            for bond in match.residue_definition.bonds:
                atom1 = match.canonical_atom_name_to_index.get(bond.atom1)
                atom2 = match.canonical_atom_name_to_index.get(bond.atom2)
                if atom1 is not None and atom2 is not None:
                    idcs = sort_tuple((atom1, atom2))
                    if idcs in bonds:
                        assert bonds[idcs].order == bond.order
                    bonds[idcs] = bond
            for idcs, bond in (
                (match.posterior_bond_idcs, match.residue_definition.linking_bond),
                (match.prior_bond_idcs, match.residue_definition.linking_bond),
                (match.crosslink_idcs, match.residue_definition.crosslink),
            ):
                if idcs is not None:
                    assert bond is not None
                    idcs = sort_tuple(idcs)
                    if idcs in bonds:
                        assert bonds[idcs].order == bond.order
                    bonds[idcs] = bond

        # We should now have added every atom in the PDB file's first model
        if len(pdb_idcs) != 0:
            raise PabloError(
                f"Unidentified atoms: {pdb_idcs} (lines {', '.join(str(self.line_no[i]) for i in pdb_idcs)})",
            )

        logging.debug("Begin adding atoms to RDMol")

        # Add the atoms to the rdmol in PDB index order
        idx_pdb_to_rdmol: dict[int, int] = {}
        for i, (atom, match) in sorted(atoms.items(), key=lambda t: t[0]):
            idx_pdb_to_rdmol[i] = rdmol.add_atom_with(
                element=atom.symbol,
                formal_charge=atom.charge,
                is_aromatic=atom.aromatic,
                properties={
                    **self._generate_atom_metadata(i),
                    "_name": atom.name if use_canonical_names else self.name[i],
                    "canonical_name": atom.name,
                    "matched_residue_description": match.residue_definition.description,
                    "matched_stereo": atom.stereo or "",
                },
            )

        # rdmol indices should be in the same order as PDB indices
        assert (  # nofmt
            sorted(idx_pdb_to_rdmol.values())
            == [idx_pdb_to_rdmol[k] for k in sorted(idx_pdb_to_rdmol.keys())]
        )

        logging.debug("Begin adding bonds to RDMol")

        # Add the bonds to the rdmol
        for (atom1, atom2), bond in bonds.items():
            logging.debug(f"{atom1, atom2, bond.order}")
            rdmol.add_bond(idx_pdb_to_rdmol[atom1], idx_pdb_to_rdmol[atom2], bond.order)

        logging.debug("Sanitizing...")
        # Sanitize and apply edits
        rdmol = rdmol.sanitize_and().freeze()

        logging.debug("Checking radicals")

        # Check for radicals to give more detailed error reporting
        def format_atom(atom: RdAtom) -> str:
            return (
                f"{atom.properties['chain_id']}:{atom.properties['residue_name']}"
                + f"{atom.properties['res_seq']}.{atom.name}"
                + f" (l{atom.properties['pdb_line_no']})"
            )

        for atom in rdmol.atoms:
            if atom.n_radical_electrons != 0:
                logging.warning(
                    f"Atom {format_atom(atom)} has {atom.n_radical_electrons} radical"
                    + f" electrons, formal charge {atom.formal_charge:+}, and"
                    + f" {atom.n_bonds} bonds.",
                )
                for bond in atom.bonds:
                    logging.warning(
                        f"  {format_atom(bond.begin_atom)} bonded to {format_atom(bond.end_atom)} with order {bond.order}",
                    )

        logging.debug("Finished producing rdmol")
        return rdmol

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Get a dictionary of all data fields for a specific atom index.

        Parameters
        ----------
        index
            The index of the atom to get data for
        """
        return {
            field.name: getattr(self, field.name)[index]
            for field in dataclasses.fields(self)
        }

    def _generate_atom_metadata(
        self,
        pdb_index: int,
    ) -> dict[str, str | int]:
        """
        Generate a metadata dict for an atom at the specified PDB index.

        The returned dictionary is used to set the atom's ``metadata``
        attribute.

        Parameters
        ----------
        pdb_index
            The PDB index of the atom
        """
        if self.res_idx is None:
            list(self.residue_indices)
            assert self.res_idx is not None, (
                "Iterating over residue indices sets res_idx"
            )

        try:
            res_number = dec_hex(self.res_seq[pdb_index])
        except ValueError:
            res_number = self.res_idx[pdb_index]

        return {
            "residue_name": self.res_name[pdb_index],
            "residue_number": res_number,
            "res_seq": self.res_seq[pdb_index],
            "residue_index": self.res_idx[pdb_index],
            "insertion_code": self.i_code[pdb_index],
            "chain_id": self.chain_id[pdb_index],
            "pdb_index": pdb_index,
            "atom_serial": self.serial[pdb_index],
            "b_factor": str(self.temp_factor[pdb_index]),
            "occupancy": str(self.occupancy[pdb_index]),
            "alt_loc": str(self.alt_loc[pdb_index]),
            "pdb_line_no": self.line_no[pdb_index],
            "used_synonym": self.name[pdb_index],
        }
