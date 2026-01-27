from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Literal, Self, overload
from urllib.error import URLError
from urllib.request import HTTPError, urlopen

from openff.pablo._cif import cif_ints, cif_str, cif_strs, parse_cif
from openff.pablo._utils import flatten, unwrap
from openff.pablo.exceptions import PabloError

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


class CcdCache(Mapping[str, tuple[ResidueDefinition, ...]]):
    """
    Caches, patches, and presents the CCD as a Python ``Mapping``.

    This class is a wrapper around a ``dict`` that stores residue definitions.
    When a residue is requested via the indexing syntax (for example,
    ``my_ccd_cache["ALA"]``), this dictionary is checked first. If the residue
    is not present and the ``auto_download`` attribute is truthy, the CCD is
    then checked. If the residue cannot be retrieved from the inner ``dict`` or
    the CCD, a ``KeyError`` is raised.

    Iterating over the mapping, checking its length, using the ``in`` operator,
    or otherwise treating the ``CcdCache`` as a mapping other than with the
    indexing syntax works only on the inner ``dict``. As a result, accessing
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
        addition to ``library_paths``. If ``None``, no on-disk cache is used,
        though the inner dictionary still functions as an in-memory cache.
    preload
        A list of residue names to ensure are present when initializing the
        class. If absent from the library or cache directory, these are
        downloaded from the CCD even if ``auto_download`` is falsy. Note that
        a download failure will raise an error when the corresponding residue is
        first requested, not at instantiation.
    patches
        Functions to call on ``ResidueDefinitions`` downloaded from the CCD
        before they are returned or added to the inner ``dict``. An iterable of
        maps from residue names each to a single callable. Each map is applied
        to residues with the given name in the order they are iterated over. Any
        patches corresponding to key ``"*"`` will be applied to all residues
        before the more specific patches in its map.
    extra_definitions
        Additional residue definitions to add to the cache. Note that patches
        are not applied to these definitions.
    auto_download
        If ``True``, automatically attempt to download unknown residues from the
        CCD. If ``False``, raise ``KeyError`` when residues absent from the
        cache are requested.
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
        cache_path: Path | str | None = None,
        preload: list[str] = [],
        patches: Iterable[
            Mapping[
                str,
                Callable[[ResidueDefinition], list[ResidueDefinition]],
            ]
        ] = {},
        extra_definitions: Mapping[str, Iterable[ResidueDefinition]] = {},
        auto_download: bool = False,
    ):
        self.auto_download = auto_download

        if cache_path is None:
            self._cache_path = None
        else:
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
                self.get_from_ccd(resname)
            except Exception:
                # If a preload fails, skip it - we want an error at runtime, not importtime
                pass

        for resname, resdefs in extra_definitions.items():
            self._add_definitions(resdefs, resname)

    def __getitem__(self, key: str) -> tuple[ResidueDefinition, ...]:
        res_name = key.upper()
        if res_name in ["UNK", "UNL"] and res_name not in self._definitions:
            # These residue names are reserved for unknown ligands/peptide residues
            raise KeyError(res_name, "reserved residue name")
        # Check the loaded definitions
        try:
            return tuple(self._definitions[res_name])
        except KeyError:
            pass

        # Don't check the library; it was loaded in __init__, and
        # trying to hot-load a definition that's already in self._definitions
        # would be very confusing.

        # If it's a residue id that could be in the CCD, try to download it.
        # We should check the cache here in case this file has been downloaded
        # since this CcdCache was created - say, in a separate process, or
        # another CcdCache - but this is done by _add_definition_from_ccd()
        if self.auto_download:
            try:
                return tuple(self._add_definition_from_ccd(res_name))
            except HTTPError:
                raise KeyError(res_name, "unknown and absent from CCD")
            except URLError:
                raise KeyError(res_name, "unknown and CCD could not be accessed")
        else:
            raise KeyError(res_name, "unknown and auto_download=False")

    def get_from_ccd(
        self,
        res_name: str,
        patch: bool = True,
    ) -> list[ResidueDefinition]:
        """
        Retrieve a residue definition from the CCD, adding it to the cache.

        Downloads the residue definition from the CCD, applies any patches, adds
        it to the cache, and returns the new, patched residue definitions. Does
        not return any definitions existing in the cache. Always downloads, even
        when the entry is already in the cache or ``auto_download`` is falsy.

        Parameters
        ==========
        res_name
            The 3-letter code for the residue.
        patch
            If ``True``, apply patches before adding to the cache and returning.
            If ``False``, do not apply patches.
        """
        s = self._download_cif(res_name)
        return self._add_definition_from_str(
            s,
            res_name=res_name,
            patch=patch,
            return_old_definitions=False,
        )

    def _add_definition_from_ccd(self, res_name: str) -> list[ResidueDefinition]:
        """Add a definition by retrieving it from the CCD (or the cache)"""
        if self._cache_path is None:
            s = self._download_cif(res_name)
        else:
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
        assert residue_definition.residue_name is not None

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
        return_old_definitions: bool = True,
    ) -> list[ResidueDefinition]:
        definition = self._res_def_from_ccd_str(s)
        if res_name is None:
            if definition.residue_name is None:
                raise PabloError(
                    "Anonymous residue definitions cannot be added to a CcdCache",
                )
            res_name = definition.residue_name.upper()

        return self._add_definitions(
            (definition,),
            res_name,
            patch=patch,
            return_old_definitions=return_old_definitions,
        )

    def _add_definitions(
        self,
        definitions: Iterable[ResidueDefinition],
        res_name: str,
        patch: bool = False,
        return_old_definitions: bool = True,
    ) -> list[ResidueDefinition]:
        if patch:
            definitions = list(flatten(map(self._apply_patches, definitions)))
        else:
            definitions = list(definitions)

        for definition in definitions:
            if definition.residue_name is None:
                raise PabloError(
                    "Anonymous residue definitions cannot be added to a CcdCache",
                )
            if res_name != definition.residue_name.upper():
                raise PabloError(
                    f"ResidueDefinition {definition.residue_name}"
                    + f" ({definition.description}) must have residue name {res_name}",
                )

        stored_definitions = self._definitions.setdefault(res_name, [])
        # Using a dict comprehension here rather than simple
        # set(definitions) - set(stored_definitions) to preserve ordering
        old_definitions = set(stored_definitions)
        stored_definitions.extend(
            {k: None for k in definitions if k not in old_definitions},
        )
        if return_old_definitions:
            return stored_definitions
        else:
            return definitions

    def _download_cif(self, resname: str) -> str:
        with urlopen(
            f"https://files.rcsb.org/ligands/download/{resname.upper()}.cif",
        ) as stream:
            s: str = stream.read().decode("utf-8")
        if self._cache_path is not None:
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
            *((self._cache_path,) if (cached and self._cache_path is not None) else ()),
            *(self._library_paths if library else ()),
        }:
            yield from path.glob(pattern)

    @staticmethod
    def _res_def_from_ccd_str(s: str) -> ResidueDefinition:
        # TODO: Handle residues like CL with a single atom properly (no tables)
        data = parse_cif(s)
        block = data[0]

        residueName = cif_str(unwrap(block["_chem_comp.id"])).upper()
        residue_description = cif_str(unwrap(block["_chem_comp.name"]))
        linking_type = cif_str(unwrap(block["_chem_comp.type"]))
        linking_bond = LINKING_TYPES[linking_type]

        atoms = [
            AtomDefinition(
                name=atom_name,
                synonyms=tuple(
                    [alt_atom_name] if alt_atom_name != atom_name else [],
                ),
                symbol=symbol[0:1].upper() + symbol[1:].lower(),
                leaving=leaving_flag == "Y",
                charge=charge,
                aromatic=aromatic_flag == "Y",
                stereo=None if stereo == "N" else stereo,  # pyright: ignore[reportArgumentType]
            )
            for atom_name, alt_atom_name, symbol, leaving_flag, (
                charge,
                aromatic_flag,
                stereo,
            ) in zip(
                cif_strs(block["_chem_comp_atom.atom_id"]),
                cif_strs(block["_chem_comp_atom.alt_atom_id"]),
                cif_strs(block["_chem_comp_atom.type_symbol"]),
                cif_strs(block["_chem_comp_atom.pdbx_leaving_atom_flag"]),
                zip(  # Keep zip invocations to <=5 arguments to keep types attached
                    cif_ints(block["_chem_comp_atom.charge"]),
                    cif_strs(block["_chem_comp_atom.pdbx_aromatic_flag"]),
                    cif_strs(block["_chem_comp_atom.pdbx_stereo_config"]),
                    strict=True,
                ),
                strict=True,
            )
        ]

        # If chem_comp_bond.atom_id_1 is missing, we need to make sure the
        # entire chem_comp_bond loop is missing before we conclude that there
        # are no bonds. If the loop is present, we need to raise an error.
        # Typically, chem_comp_bond.atom_id_1 will be present,
        # and the `or` will short circuit, or else the loop will be missing,
        # in which case the `any` has nothing to do.
        if "_chem_comp_bond.atom_id_1" in block or any(
            key.startswith("_chem_comp_bond.") for key in block.keys()
        ):
            bonds = [
                BondDefinition(
                    atom1=atom1,
                    atom2=atom2,
                    order={"SING": 1, "DOUB": 2, "TRIP": 3, "QUAD": 4}[order],
                    aromatic=aromatic_flag == "Y",
                    stereo=None if stereo_flag == "N" else stereo_flag,  # pyright: ignore[reportArgumentType]
                )
                for atom1, atom2, order, aromatic_flag, stereo_flag in zip(
                    cif_strs(block["_chem_comp_bond.atom_id_1"]),
                    cif_strs(block["_chem_comp_bond.atom_id_2"]),
                    cif_strs(block["_chem_comp_bond.value_order"]),
                    cif_strs(block["_chem_comp_bond.pdbx_aromatic_flag"]),
                    cif_strs(block["_chem_comp_bond.pdbx_stereo_config"]),
                    strict=True,
                )
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

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, str):
            raise TypeError(
                f"CcdCache keys are residue names of type str, not {type(value)}",
            )
        return value in self._definitions

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
        Get a copy of this ``CcdCache`` with additional definitions added.

        Definitions may be supplied as a mapping from residue names to sequences
        of residue definitions, or as a sequence of residue definitions. In the
        latter case, the residue names are taken from the residue definitions
        themselves.

        Note that patches are not applied to the new definitions.

        Examples
        ========

        Add a custom definition to the ``STD_CCD_CACHE``. We use
        a 4-letter residue code as they are supported by Pablo's PDB reader and
        do not clash with the CCD's definitions.

        >>> from openff.pablo import STD_CCD_CACHE, ResidueDefinition
        >>> my_ccd_cache = STD_CCD_CACHE.with_([
        ...     ResidueDefinition.from_smiles(
        ...         "[H:1][O:2][O:3][H:4]",
        ...         {1: "H1", 2: "O1", 3: "O2", 4: "H2"},
        ...         "HOOH",
        ...     )
        ... ])

        Add protonation variants of a residue by specifying acidic and basic
        atoms.

        >>> from openff.pablo import STD_CCD_CACHE, ResidueDefinition
        >>>
        >>> # Get the GABA (γ-amino butanoic acid) residue definition from CCD
        >>> gaba_resdef = STD_CCD_CACHE["ABU"][0]
        >>>
        >>> # Generate the variants and add them to a new cache
        >>> my_ccd_cache = STD_CCD_CACHE.with_({
        ...     "ABU": gaba_resdef.vary_protonation(
        ...         acidic=["HXT"], # Atom name of abstractable proton
        ...         basic=[("N", "H3")], # Atom to protonate, name of new proton
        ...     )[1:], # Skip the first entry, which is already in the cache
        ... })
        >>> # Should have added three variants - positive, negative, zwitterion
        >>> len(my_ccd_cache["ABU"]) - len(STD_CCD_CACHE["ABU"])
        3

        See Also
        ========
        with_replaced, without

        """
        if not isinstance(definitions, Mapping):
            definitions_map: dict[str, list[ResidueDefinition]] = {}
            for resdef in definitions:
                if resdef.residue_name is None:
                    raise PabloError(
                        "Anonymous residue definitions cannot be added to a CcdCache",
                    )
                definitions_map.setdefault(resdef.residue_name, []).append(resdef)
            definitions = definitions_map

        new = deepcopy(self)
        for resname, resdefs in definitions.items():
            new._add_definitions(resdefs, resname)
        return new

    def with_replaced(
        self,
        definitions: Mapping[str, Sequence[ResidueDefinition]]
        | Sequence[ResidueDefinition],
    ) -> Self:
        """
        Get a copy of this ``CcdCache`` with some definitions replaced.

        Similar to ``with_``, but does not retain existing definitions for the
        specified residue names. All residue names that are keys of a
        ``definitions`` mapping or are residue names in a ``definitions``
        sequence are removed from the new ``CcdCache`` before adding the
        new definitions.

        Note that patches are not applied to the new definitions.

        See Also
        ========
        with_, without

        """
        if not isinstance(definitions, Mapping):
            definitions_map: dict[str, list[ResidueDefinition]] = {}
            for resdef in definitions:
                if resdef.residue_name is None:
                    raise PabloError(
                        "Anonymous residue definitions cannot be added to a CcdCache",
                    )
                definitions_map.setdefault(resdef.residue_name, []).append(resdef)
            definitions = definitions_map

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
        Get a copy of this ``CcdCache`` lacking any definitions with some names.

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
        """
        Add a patch to the residues loaded via a copy of this ``CcdCache``.

        The patch is added to a copy of the ``CcdCache``, and the copy is
        returned. The original ``CcdCache`` is left unmodified.

        The patch function is called on each residue definition stored under the
        given residue name. The returned residue definitions are concatenated
        and replace the originals. Patches can therefore add, modify, split, or
        replace residue definitions depending on whether they include the
        original definition in the output.

        The patch is applied to all definitions in the cache when this function
        is applied, as well as any definitions downloaded from the CCD in the
        future. It is not applied to definitions added by the other
        ``CcdCache.with_*()`` methods.
        """
        new = deepcopy(self)
        new._patches.append({residue_name: patch})
        if residue_name in new._definitions:
            new._definitions[residue_name] = list(
                flatten(patch(resdef) for resdef in new._definitions[residue_name]),
            )
        return new

    @overload
    def with_crosslink(
        self,
        *,
        residues: tuple[str],
        linking_atoms: tuple[str],
        leaving_atoms: tuple[Collection[str]],
        bond_order: int,
        aromatic: bool = False,
        stereo: Literal["E", "Z"] | None = None,
    ) -> Self: ...

    @overload
    def with_crosslink(
        self,
        *,
        residues: tuple[str, str],
        linking_atoms: tuple[str, str],
        leaving_atoms: tuple[Collection[str], Collection[str]],
        bond_order: int,
        aromatic: bool = False,
        stereo: Literal["E", "Z"] | None = None,
    ) -> Self: ...

    def with_crosslink(
        self,
        *,
        residues: tuple[str, ...],
        linking_atoms: tuple[str, ...],
        leaving_atoms: tuple[Collection[str], ...],
        bond_order: int = 1,
        aromatic: bool = False,
        stereo: Literal["E", "Z"] | None = None,
    ) -> Self:
        """
        Add a custom crosslink between residues.

        Parameters
        ==========
        residues
            The names of the residues the crosslink should be formed between.
            May have 1 entry for a homodimer or 2 entries for a heterodimer.
        linking_atoms
            The atom names between which the crosslink bond is formed. Should
            have one corresponding entry for each residue.
        leaving_atoms
            The atom names that are absent from the PDB file when the crosslink
            is in place. These atom names are replaced by the crosslink bond.
            Should have 1 entry for a homodimer or 2 entries for a heterodimer.
            Each entry should include all atom names that are absent when the
            crosslink exists. For example, for a disulfide bond between two
            cysteines, the leaving atoms are ``[["HG"]]``. For a peptide bond
            between two alpha amino acids, the leaving atoms are
            ``[["H2"], ["OXT", "HXT"]]``, though peptide bonds are typically
            modelled with linking bonds rather than crosslinks.
        bond_order
            The bond order of the crosslink; 1 for a single bond, 2 for a double
            bond, etc.
        aromatic
            ``True`` if the bond is aromatic; ``False`` otherwise.
        stereo
            The stereochemistry of the crosslink bond. For a bond without
            stereochemistry, choose ``None``.
        """
        if len(residues) != len(linking_atoms):
            raise PabloError(
                "residues and linking_atoms arguments should have same number of entries",
            )
        if len(residues) != len(leaving_atoms):
            raise PabloError(
                "residues and leaving_atoms arguments should have same number of entries",
            )

        if len(residues) == 1:
            res_a, res_b = residues[0], residues[0]
            link_a, link_b = linking_atoms[0], linking_atoms[0]
            leavers_a, leavers_b = leaving_atoms[0], leaving_atoms[0]
        elif len(residues) == 2:
            res_a, res_b = residues
            link_a, link_b = linking_atoms
            leavers_a, leavers_b = leaving_atoms
        else:
            raise PabloError("residues argument should have 1 or 2 entries")

        bond_a = BondDefinition(
            atom1=link_a,
            atom2=link_b,
            order=bond_order,
            aromatic=aromatic,
            stereo=stereo,
        )
        bond_b = bond_a.flipped()

        modified_residues: list[ResidueDefinition] = []
        for res_name, bond, leavers in (
            (res_a, bond_a, leavers_a),
            (res_b, bond_b, leavers_b),
        ):
            for residue in self[res_name]:
                if residue.crosslink is not None:
                    raise PabloError(
                        f"{residue.residue_name} not supported: existing crosslink",
                    )

                modified_residues.append(
                    residue.replace(
                        crosslink=bond,
                        atoms=tuple(
                            (
                                atom.replace(leaving=True)
                                if atom.name in leavers
                                else atom
                            )
                            for atom in residue.atoms
                        ),
                    ),
                )
        return self.with_replaced(modified_residues)

    def with_virtual_sites(
        self,
        residue_name: str,
        virtual_sites: Iterable[str],
    ) -> Self:
        """
        Copy ``self``, adding new residue definitions requiring some virtual sites.

        The new definition is added to a copy of the ``CcdCache``, and the copy
        is returned. The original ``CcdCache`` is left unmodified.

        A new residue definition is added for each definition currently stored
        in the cache under the given name. The new definition requires that all
        the given virtual site names be present in order for it to match, and it
        discards the corresponding ATOM/HETATM records.

        This method works by adding a patch. It will affect any residue
        definition already added to the cache under the given name, or any
        definition downloaded in the future, but not any definition added in the
        future by the ``with_`` or ``with_replaced`` methods.
        """
        virtual_sites = tuple(virtual_sites)
        return self.with_patch(
            residue_name,
            lambda resdef: [resdef, resdef.replace(virtual_sites=virtual_sites)],
        )

    def with_vsite_water(self) -> Self:
        """
        Copy ``self``, adding new definitions for common multisite water models.

        The new definitions are added to a copy of the ``CcdCache``, and the
        copy is returned. The original ``CcdCache`` is left unmodified.

        The new definitions require that all the virtual site names be
        present in order for them to match, and they discard the corresponding
        ATOM/HETATM records. The name for the 4-point model virtual site is
        ``EPW``, and for the 5-point model ``EP1`` and ``EP2``.

        This method works by adding a patch. It will affect any 3-atom residue
        definitions already added to the cache under the names ``HOH``, ``WAT``,
        or ``SOL``, or any so-named definition downloaded in the future, but not
        any  definition added in the future by the ``with_`` or ``with_replaced``
        methods.
        """

        def add_vsites_to_water(resdef: ResidueDefinition) -> list[ResidueDefinition]:
            if resdef.n_expected_atoms == 3 and {
                atom.name for atom in resdef.atoms
            } == {"O", "H1", "H2"}:
                return [
                    resdef,
                    resdef.replace(virtual_sites=["EPW"]),
                    resdef.replace(virtual_sites=["EP1", "EP2"]),
                ]
            else:
                return [resdef]

        return (
            self.with_patch(
                "HOH",
                add_vsites_to_water,
            )
            .with_patch(
                "WAT",
                add_vsites_to_water,
            )
            .with_patch(
                "SOL",
                add_vsites_to_water,
            )
        )

    def with_varied_protonation(
        self,
        residue_name: str,
        *,
        acidic: Iterable[str] = (),
        basic: Iterable[tuple[str, str]] = (),
    ) -> Self:
        """
        Get a copy of ``self`` with all combinations of some protonation states.

        Note that all combinations of protonations and deprotonations are
        generated; this means that if ``acidic`` has length ``n`` and ``basic``
        has length ``m``, ``2**(n+m)`` variants will be generated for each
        existing variant.

        If no variants at all are generated, ``PabloError`` is raised. Otherwise,
        whatever variants make sense are created for each existing variant.

        This method will download the given residue name from the CCD if it is
        not already in the cache.

        Parameters
        ==========
        residue_name
            The name of the residue to generate alternate protonation states
            for.
        acidic
            Existing hydrogen atoms that can be removed to form a new
            protonation state. Each element specifies an atom name to remove,
            decrementing the formal charge on the neighbouring heavy atom.
            Multiply bonded, unbonded, missing, or non-hydrogen atoms are
            skipped unless no variants at all are generated.
        basic
            Existing non-hydrogen atoms that can be protonated to form a new
            protonation state, as well as the canonical name of the new
            hydrogen. Each tuple specifies an atom name to protonate (increment
            the formal charge and form a bond) and the name of the added proton.
            Unknown heavy atoms and new atom names that clash with existing
            names raise are skipped unless no variants at all are generated.
        """
        original_resdefs = self[residue_name]
        if len(original_resdefs) == 0:
            raise PabloError("cannot (de)protonate unknown residue {residue_name}")

        new_resdefs: list[ResidueDefinition] = []
        for resdef in original_resdefs:
            resdef_variants = resdef.vary_protonation(
                acidic=acidic,
                basic=basic,
                skip_errors=True,
            )
            # Only add the new variants so we can keep count of how many we've
            # come up with
            for i, variant in enumerate(resdef_variants[1:]):
                # Filter out duplicate variants
                if not any(
                    variant._is_isomorphic_to(other) for other in resdef_variants[:i]
                ):
                    new_resdefs.append(variant)
        if len(new_resdefs) == 0:
            raise PabloError("no new protonation variants added")

        return self.with_({residue_name: new_resdefs})


# TODO: Fill in this data
LINKING_TYPES: dict[str, BondDefinition | None] = {
    # "D-beta-peptide, C-gamma linking".upper(): [],
    # "D-gamma-peptide, C-delta linking".upper(): [],
    # "D-peptide COOH carboxy terminus".upper(): [],
    # "D-peptide NH3 amino terminus".upper(): [],
    "D-peptide linking".upper(): PEPTIDE_BOND,
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
