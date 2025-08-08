from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Self, no_type_check
from urllib.error import URLError
from urllib.request import HTTPError, urlopen

import xdg.BaseDirectory as xdg_base_dir
from openmm.app.internal.pdbx.reader.PdbxReader import PdbxReader

from openff.pablo._utils import flatten

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

    This class is a wrapper around a ``dict`` that stores residue definitions.
    When a residue is requested via the indexing syntax (for example,
    ``my_ccd_cache["ALA"]``) or the ``in`` operator, this dictionary is checked
    first. If the residue is not present, the CCD is then checked. If the
    residue cannot be retrieved from the inner ``dict`` or the CCD, a
    ``KeyError`` is raised or ``False`` is returned as appropriate.

    Iterating over the mapping, checking its length, or otherwise treating the
    ``CcdCache`` as a mapping other than with the indexing syntax or ``in``
    operator works only on the inner ``dict``. As a result, accessing
    a residue via indexing may return a value even if these other methods
    suggest it won't.

    ``CcdCache`` can apply patches to the entries it downloads from the CCD.
    This is used to work around known errors, deficiencies and inconsistencies
    in the CCD definitions. Patches are specified as functions that take a
    single residue definition and return a list of them.

    The ``extra_definitions`` and the ``with_()`` and ``with_replaced`` methods
    allow custom definitions to be added to a ``CcdCache``. These custom
    definitions are not patched. Since they are stored alongside cached entries
    in the inner dictionary, custom definitions supercede any that have not
    already been downloaded from the CCD.

    When a CCD entry is downloaded, the corresponding CIF file is stored in the
    ``cache_path``. This means that each entry will be downloaded only once,
    even across multiple Python invocations. All entries in the cache are loaded
    (and patches applied) when a new ``CcdCache`` is created. ``CcdCache``
    assumes that the files in the cache path were downloaded from the CCD and
    may do unexpected things if they are edited by hand.

    Users may provide additional CCD entries by specifying library paths. By
    default, this is used to ship commonly used residues with Pablo. At the
    moment, patches are applied to files in library paths, but it is likely that
    in the future they won't be and residues shipped with Pablo will be shipped
    pre-patched to speed up load times. Like with the cache, files from library
    paths are loaded when a ``CcdCache`` is created.

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
        addition to ``library_paths``.
    preload
        A list of residue names to download when initializing the class.
    patches
        Functions to call on ``ResidueDefinitions`` downloaded from the CCD
        before they are returned or added to the inner `dict``. An iterable of
        maps from residue names each to a single callable. Each map is applied
        to residues with the given name in the order they are iterated over. Any
        patches corresponding to key ``"*"`` will be applied to all residues
        before the more specific patches in its map.
    extra_definitions
        Additional residue definitions to add to the cache. Note that patches
        are not applied to these definitions.
    """

    # TODO: remove library_paths in favour of loading pre-patched entries
    #       distributed with Pablo via extra_definitions

    # TODO: Consider storing extra definitions and patched CCD entries in
    #       separate inner dicts to make behaviour less dependent on access
    #       sequencing

    # TODO: Make cache_path optional

    # TODO: Compress the on-disk cache

    def __init__(
        self,
        library_paths: Iterable[Path | str],
        cache_path: Path | str = Path(
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
        self._cache_path = Path(cache_path).resolve()
        self._cache_path.mkdir(parents=True, exist_ok=True)
        self._library_paths = {Path(path).resolve() for path in library_paths}

        self._definitions: dict[str, list[ResidueDefinition]] = {}
        self._patches: list[
            dict[
                str,
                Callable[[ResidueDefinition], list[ResidueDefinition]],
            ]
        ] = [dict(d) for d in patches]

        for path in self._glob("*.cif"):
            try:
                self._add_definition_from_str(path.read_text(), patch=True)
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

        # Don't check the library; it was loaded in __init__, and
        # trying to hot-load a definition that's already in self._definitions
        # would be very confusing.

        # If it's a residue id that could be in the CCD, try to download it.
        # We should check the cache here in case this file has been downloaded
        # since this CcdCache was created - say, in a separate process, or
        # another CcdCache - but this is done by _add_definition_from_ccd()
        try:
            return self._add_definition_from_ccd(res_name)
        except HTTPError:
            raise KeyError(res_name, "unknown and absent from CCD")
        except URLError:
            raise KeyError(res_name, "unknown and CCD could not be accessed")

    def _add_definition_from_ccd(self, res_name: str) -> list[ResidueDefinition]:
        """Add a definition by retrieving it from the CCD (or the cache)"""
        cache_path = self._cache_path / f"{res_name}.cif"
        if cache_path.is_file():
            s = cache_path.read_text()
        else:
            s = self._download_cif(res_name)

        return self._add_definition_from_str(s, res_name=res_name, patch=True)

    def _apply_patches(
        self,
        residue_definition: ResidueDefinition,
    ) -> list[ResidueDefinition]:
        with _skip_residue_definition_validation():
            definitions: list[ResidueDefinition] = [residue_definition]
            for patch_dict in self._patches:
                star_patch = patch_dict.get("*", lambda x: [x])
                res_patch = patch_dict.get(
                    residue_definition.residue_name.upper(),
                    lambda x: [x],
                )
                patched_definitions: list[ResidueDefinition] = []
                for definition in definitions:
                    for star_patched_res in star_patch(definition):
                        patched_definitions.extend(res_patch(star_patched_res))
                definitions = patched_definitions

        for definition in definitions:
            definition._validate()

        return definitions

    def _add_definition_from_str(
        self,
        s: str,
        res_name: str | None = None,
        patch: bool = False,
    ) -> list[ResidueDefinition]:
        definition = self._res_def_from_ccd_str(s)
        if res_name is None:
            res_name = definition.residue_name.upper()

        return self._add_definitions((definition,), res_name, patch=patch)

    def _add_definitions(
        self,
        definitions: Iterable[ResidueDefinition],
        res_name: str,
        patch: bool = False,
    ) -> list[ResidueDefinition]:
        definitions = list(definitions)
        for definition in definitions:
            if res_name != definition.residue_name.upper():
                raise ValueError(
                    f"ResidueDefinition {definition.residue_name}"
                    + f" ({definition.description}) must have residue name {res_name}",
                )
        if patch:
            definitions = flatten(map(self._apply_patches, definitions))

        stored_definitions = self._definitions.setdefault(res_name, [])
        # Using a dict comprehension here rather than simple
        # set(definitions) - set(stored_definitions) to preserve ordering
        old_definitions = set(stored_definitions)
        stored_definitions.extend(
            {k: None for k in definitions if k not in old_definitions},
        )
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
                    virtual_sites=(),
                )

            return residue_definition

        return inner(s)

    def __contains__(self, value: object) -> bool:
        if value in self._definitions:
            return True
        if not isinstance(value, str):
            raise TypeError(
                f"CcdCache keys are residue names of type str, not {type(value)}",
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

        Note that patches are not applied to the new definitions.

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

        See Also
        ========
        with_replaced, without

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

    def with_replaced(
        self,
        definitions: Mapping[str, Sequence[ResidueDefinition]],
    ) -> Self:
        """
        Get a new ``CcdCache`` with replaced definitions of some residue names.

        Similar to ``with_``, but does not retain existing definitions for the
        specified residue names. All residue names that are keys of
        ``definitions`` are removed from the new ``CcdCache`` before adding the
        new definitions.

        Note that patches are not applied to the new definitions.

        See Also
        ========
        with_, without

        """
        new = deepcopy(self)
        for resname, resdefs in definitions.items():
            del new._definitions[resname]
            new._add_definitions(resdefs, resname)
        return new

    def without(
        self,
        residue_names: Iterable[str],
    ) -> Self:
        """
        Get a new ``CcdCache`` with some residue names removed.

        All definitions for each of the given residue names will not be present
        in the new cache. Note that residues that are in the CCD will still be
        returned when they are requested, as long as they can be re-downloaded
        or found in the cache.

        See Also
        ========
        with_replaced, with_

        """
        new = deepcopy(self)
        for resname in residue_names:
            del new._definitions[resname]
        return new

    def with_patch(
        self,
        residue_name: str,
        patch: Callable[[ResidueDefinition], list[ResidueDefinition]],
    ) -> Self:
        self._patches.append({residue_name: patch})
        if residue_name in self._definitions:
            self._definitions[residue_name] = list(
                flatten(patch(resdef) for resdef in self._definitions[residue_name]),
            )
        return self

    def with_virtual_sites(
        self,
        residue_name: str,
        virtual_sites: Iterable[str],
    ) -> Self:
        virtual_sites = tuple(virtual_sites)
        return self.with_patch(
            residue_name,
            lambda resdef: [resdef, resdef.replace(virtual_sites=virtual_sites)],
        )

    def with_vsite_water(self) -> Self:
        return self.with_patch(
            "HOH",
            lambda resdef: [
                resdef,
                resdef.replace(virtual_sites=["EPW"]),
                resdef.replace(virtual_sites=["EPW1", "EPW2"]),
            ],
        )


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
