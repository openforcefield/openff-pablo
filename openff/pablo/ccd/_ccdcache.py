from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Self, no_type_check
from urllib.error import URLError
from urllib.request import HTTPError, urlopen

import xdg.BaseDirectory as xdg_base_dir
from openmm.app.internal.pdbx.reader.PdbxReader import PdbxReader

from ..chem import PEPTIDE_BOND, PHOSPHODIESTER_BOND
from ..residue import (
    AtomDefinition,
    BondDefinition,
    ResidueDefinition,
    _skip_residue_definition_validation,
)

__all__ = [
    "CcdCache",
]


class CcdCache(Mapping[str, list[ResidueDefinition]]):
    """
    Caches, patches, and presents the CCD as a Python ``Mapping``.

    Accessing the CCD requires internet access. Without internet access, entries
    from the cache or library paths can still be loaded, as can any entries
    added to an instance of this class.

    Parameters
    ==========
    library_paths
        Paths to search for user-provided or packaged CCD entries. All paths are
        searched.
    cache_path
        The path to which to download CCD entries. This path is searched in
        addition to ``library_paths``. There is no need to include this path
        in ``library_paths``, but it doesn't hurt if you do.
    preload
        A list of residue names to download when initializing the class.
    patches
        Functions to call on the given ``ResidueDefinitions`` before they are
        returned. An iterable of maps from residue names each to a single
        callable. Each map is applied to residues with the given name in the
        order they are iterated over. The patch corresponding to key ``"*"``
        will be applied to all residues before the more specific patches in its
        map.
    extra_definitions
        Additional residue definitions to add to the cache. Note that patches
        are not applied to these definitions.
    """

    # TODO: Methods for adding entries from mapped SMILES

    def __init__(
        self,
        library_paths: Iterable[Path],
        cache_path: Path = Path(
            xdg_base_dir.save_cache_path("openff-pablo"),
            "ccd_cache",
        ),
        preload: list[str] = [],
        patches: Iterable[
            Mapping[
                str,
                Callable[[ResidueDefinition], list[ResidueDefinition]],
            ]
        ] = {},
        extra_definitions: Mapping[str, Iterable[ResidueDefinition]] = {},
    ):
        self._cache_path = cache_path.resolve()
        self._cache_path.mkdir(parents=True, exist_ok=True)

        self._library_paths = {path.resolve() for path in library_paths}

        self._definitions: dict[str, list[ResidueDefinition]] = {}
        self._patches: list[
            dict[
                str,
                Callable[[ResidueDefinition], list[ResidueDefinition]],
            ]
        ] = [dict(d) for d in patches]

        for path in self._glob("*.cif"):
            try:
                self._add_definition_from_str(path.read_text())
            except Exception:
                # If adding a file fails, skip it - we want an error at runtime, not importtime
                pass

        for resname in set(preload) - set(self._definitions):
            try:
                self[resname]
            except Exception:
                # If a preload fails, skip it - we want an error at runtime, not importtime
                pass

        for resname, resdefs in extra_definitions.items():
            self._add_definitions(resdefs, resname)

    def __getitem__(self, key: str) -> list[ResidueDefinition]:
        res_name = key.upper()
        if res_name in ["UNK", "UNL"] and res_name not in self._definitions:
            # These residue names are reserved for unknown ligands/peptide residues
            raise KeyError(res_name, "reserved residue name")

        # Check the loaded definitions
        try:
            return self._definitions[res_name]
        except KeyError:
            pass

        # Don't check the library or cache; they were loaded in __init__, and
        # trying to hot-load a definition that's already in self._definitions
        # would be very confusing.

        # If it's a residue id that could be in the CCD, try to download it
        try:
            return self._add_definition_from_str(
                self._download_cif(res_name),
                res_name=res_name,
            )
        except HTTPError:
            raise KeyError(res_name, "unknown and absent from CCD")
        except URLError:
            raise KeyError(res_name, "unknown and CCD could not be accessed")

    def _apply_patches(
        self,
        residue_definition: ResidueDefinition,
    ) -> list[ResidueDefinition]:
        with _skip_residue_definition_validation():
            definitions: list[ResidueDefinition] = [residue_definition]
            for patch_dict in self._patches:
                patched_definitions: list[ResidueDefinition] = []
                for definition in definitions:
                    star_patch = patch_dict.get("*", lambda x: [x])
                    res_patch = patch_dict.get(
                        residue_definition.residue_name.upper(),
                        lambda x: [x],
                    )
                    for star_patched_res in star_patch(definition):
                        for patched_res in res_patch(star_patched_res):
                            patched_definitions.append(patched_res)
                definitions = patched_definitions

        for definition in definitions:
            definition._validate()

        return definitions

    def _add_definition_from_str(
        self,
        s: str,
        res_name: str | None = None,
    ) -> list[ResidueDefinition]:
        definition = self._res_def_from_ccd_str(s)
        return self._add_and_patch_definition(definition, res_name)

    def _add_and_patch_definition(
        self,
        definition: ResidueDefinition,
        res_name: str | None = None,
    ) -> list[ResidueDefinition]:
        if res_name is None:
            res_name = definition.residue_name.upper()

        return self._add_definitions(self._apply_patches(definition), res_name)

    def _add_definitions(
        self,
        definitions: Iterable[ResidueDefinition],
        res_name: str,
    ) -> list[ResidueDefinition]:
        for definition in definitions:
            if res_name != definition.residue_name.upper():
                raise ValueError(
                    f"ResidueDefinition {definition.residue_name}"
                    + f" ({definition.description}) must have residue name {res_name}",
                )
        stored_definitions = self._definitions.setdefault(res_name, [])
        stored_definitions.extend(definitions)
        return stored_definitions

    def _download_cif(self, resname: str) -> str:
        with urlopen(
            f"https://files.rcsb.org/ligands/download/{resname.upper()}.cif",
        ) as stream:
            s: str = stream.read().decode("utf-8")
        path = self._cache_path / f"{resname.upper()}.cif"
        path.write_text(s)
        return s

    def _glob(
        self,
        pattern: str,
        *,
        cached: bool = True,
        library: bool = True,
    ) -> Iterator[Path]:
        """
        Get paths matching the given glob pattern from the cache and/or library

        Parameters
        ==========
        pattern
            The glob to search for
        cached
            Whether to look in the cache path for the glob
        library
            Whether to look in the library paths for the glob
        """
        for path in {
            *((self._cache_path,) if cached else ()),
            *(self._library_paths if library else ()),
        }:
            yield from path.glob(pattern)

    @staticmethod
    def _res_def_from_ccd_str(s: str) -> ResidueDefinition:
        @no_type_check
        def inner(s):
            # TODO: Handle residues like CL with a single atom properly (no tables)
            data = []
            with StringIO(s) as file:
                PdbxReader(file).read(data)
            block = data[0]

            parent_residue_name = (
                block.getObj("chem_comp").getValue("mon_nstd_parent_comp_id").upper()
            )
            parent_residue_name = (
                None if parent_residue_name == "?" else parent_residue_name
            )
            residueName = block.getObj("chem_comp").getValue("id").upper()
            residue_description = block.getObj("chem_comp").getValue("name")
            linking_type = block.getObj("chem_comp").getValue("type").upper()
            linking_bond = LINKING_TYPES[linking_type]

            atomData = block.getObj("chem_comp_atom")
            atomNameCol = atomData.getAttributeIndex("atom_id")
            altAtomNameCol = atomData.getAttributeIndex("alt_atom_id")
            symbolCol = atomData.getAttributeIndex("type_symbol")
            leavingCol = atomData.getAttributeIndex("pdbx_leaving_atom_flag")
            chargeCol = atomData.getAttributeIndex("charge")
            aromaticCol = atomData.getAttributeIndex("pdbx_aromatic_flag")
            stereoCol = atomData.getAttributeIndex("pdbx_stereo_config")

            atoms = [
                AtomDefinition(
                    name=row[atomNameCol],
                    synonyms=tuple(
                        [row[altAtomNameCol]]
                        if row[altAtomNameCol] != row[atomNameCol]
                        else [],
                    ),
                    symbol=row[symbolCol][0:1].upper() + row[symbolCol][1:].lower(),
                    leaving=row[leavingCol] == "Y",
                    charge=int(row[chargeCol]),
                    aromatic=row[aromaticCol] == "Y",
                    stereo=None if row[stereoCol] == "N" else row[stereoCol],
                )
                for row in atomData.getRowList()
            ]

            bondData = block.getObj("chem_comp_bond")
            if bondData is not None:
                atom1Col = bondData.getAttributeIndex("atom_id_1")
                atom2Col = bondData.getAttributeIndex("atom_id_2")
                orderCol = bondData.getAttributeIndex("value_order")
                aromaticCol = bondData.getAttributeIndex("pdbx_aromatic_flag")
                stereoCol = bondData.getAttributeIndex("pdbx_stereo_config")
                bonds = [
                    BondDefinition(
                        atom1=row[atom1Col],
                        atom2=row[atom2Col],
                        order={"SING": 1, "DOUB": 2, "TRIP": 3, "QUAD": 4}[
                            row[orderCol]
                        ],
                        aromatic=row[aromaticCol] == "Y",
                        stereo=None if row[stereoCol] == "N" else row[stereoCol],
                    )
                    for row in bondData.getRowList()
                ]
            else:
                bonds = []

            with _skip_residue_definition_validation():
                residue_definition = ResidueDefinition(
                    residue_name=residueName,
                    description=residue_description,
                    linking_bond=linking_bond,
                    crosslink=None,
                    atoms=tuple(atoms),
                    bonds=tuple(bonds),
                )

            return residue_definition

        return inner(s)

    def __contains__(self, value: object) -> bool:
        if value in self._definitions:
            return True
        if not isinstance(value, str):
            raise TypeError(
                f"CcdCache contains residue names of type str, not {type(value)}",
            )

        try:
            self[value]
        except Exception:
            # This catches KeyError, but also failures to download the residue
            return False
        else:
            return True

    def __iter__(self) -> Iterator[str]:
        return self._definitions.__iter__()

    def __len__(self) -> int:
        return self._definitions.__len__()

    def with_(
        self,
        definitions: Mapping[str, Sequence[ResidueDefinition]]
        | Sequence[ResidueDefinition],
    ) -> Self:
        """
        Get a new ``CcdCache`` with additional definitions.

        Definitions may be supplied as a mapping from residue names to sequences
        of residue definitions, or as a sequence of residue definitions. In the
        latter case, the residue names are taken from the residue definitions
        themselves.

        Note that patches are not applied to the additional definitions.

        Examples
        ========

        Add a custom definition to the ``CCD_RESIDUE_DEFINITION_CACHE``. We use
        a 4-letter residue code as they are supported by Pablo's PDB reader and
        do not clash with the CCD's definitions.

        >>> from openff.pablo import CCD_RESIDUE_DEFINITION_CACHE, ResidueDefinition
        >>> my_ccd_cache = CCD_RESIDUE_DEFINITION_CACHE.with_([
        ...     ResidueDefinition.from_smiles(
        ...         "[H:1][O:2][O:3][H:4]",
        ...         {1: "H1", 2: "O1", 3: "O2", 4: "H2"},
        ...         "HOOH",
        ...     )
        ... ])

        Add protonation variants of a residue by specifying acidic and basic
        atoms.

        >>> from openff.pablo import CCD_RESIDUE_DEFINITION_CACHE, ResidueDefinition
        >>>
        >>> # Get the GABA (γ-amino butanoic acid) residue definition from CCD
        >>> gaba_resdef = CCD_RESIDUE_DEFINITION_CACHE["ABU"][0]
        >>>
        >>> # Generate the variants and add them to a new cache
        >>> my_ccd_cache = CCD_RESIDUE_DEFINITION_CACHE.with_({
        ...     "ABU": gaba_resdef.vary_protonation(
        ...         acidic=["HXT"], # Atom name of abstractable proton
        ...         basic=[("N", "H3")], # Atom to protonate, name of new proton
        ...     )[1:], # Skip the first entry, which is already in the cache
        ... })
        >>> # Should have added three variants - positive, negative, zwitterion
        >>> len(my_ccd_cache["ABU"]) - len(CCD_RESIDUE_DEFINITION_CACHE["ABU"])
        3

        """
        if not isinstance(definitions, Mapping):
            definitions_map: dict[str, list[ResidueDefinition]] = {}
            for resdef in definitions:
                definitions_map.setdefault(resdef.residue_name, []).append(resdef)
            definitions = definitions_map

        new = deepcopy(self)
        for resname, resdefs in definitions.items():
            new._add_definitions(resdefs, resname)
        return new


# TODO: Fill in this data
LINKING_TYPES: dict[str, BondDefinition | None] = {
    # "D-beta-peptide, C-gamma linking".upper(): [],
    # "D-gamma-peptide, C-delta linking".upper(): [],
    # "D-peptide COOH carboxy terminus".upper(): [],
    # "D-peptide NH3 amino terminus".upper(): [],
    # "D-peptide linking".upper(): [],
    # "D-saccharide".upper(): [],
    # "D-saccharide, alpha linking".upper(): [],
    # "D-saccharide, beta linking".upper(): [],
    # "DNA OH 3 prime terminus".upper(): [],
    # "DNA OH 5 prime terminus".upper(): [],
    "DNA linking".upper(): PHOSPHODIESTER_BOND,
    "L-DNA linking".upper(): PHOSPHODIESTER_BOND,
    "L-RNA linking".upper(): PHOSPHODIESTER_BOND,
    # "L-beta-peptide, C-gamma linking".upper(): [],
    # "L-gamma-peptide, C-delta linking".upper(): [],
    # "L-peptide COOH carboxy terminus".upper(): [],
    # "L-peptide NH3 amino terminus".upper(): [],
    "L-peptide linking".upper(): PEPTIDE_BOND,
    # "L-saccharide".upper(): [],
    # "L-saccharide, alpha linking".upper(): [],
    # "L-saccharide, beta linking".upper(): [],
    # "RNA OH 3 prime terminus".upper(): [],
    # "RNA OH 5 prime terminus".upper(): [],
    "RNA linking".upper(): PHOSPHODIESTER_BOND,
    "non-polymer".upper(): None,
    # "other".upper(): [],
    "peptide linking".upper(): PEPTIDE_BOND,
    "peptide-like".upper(): PEPTIDE_BOND,
    # "saccharide".upper(): [],
}
"""Map from the CCD's linking types to the bond formed between two such monomers"""
