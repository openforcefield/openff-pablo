import logging
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from typing import (
    Any,
    DefaultDict,
    Literal,
    TypeGuard,
    no_type_check,
    overload,
)

import rdkit
import rdkit.Chem
import rdkit.Chem.Draw
import rdkit.Chem.rdDepictor
from openff.toolkit import Molecule
from openff.toolkit.topology import Atom
from openff.toolkit.topology._mm_molecule import _SimpleMolecule
from openff.toolkit.topology.molecule import MoleculeLike
from openff.toolkit.utils import UndefinedStereochemistryError
from openff.toolkit.utils.exceptions import InvalidAtomMetadataError
from openff.units import unit
from pint import Quantity
from rdkit.Chem.rdChemReactions import ReactionFromSmarts

from openff.pablo.exceptions import PabloError

__all__ = [
    "default_dict",
    "unwrap",
    "sort_tuple",
    "flatten",
    "with_neighbours",
    "float_or_unknown",
    "dec_hex",
    "charge_int_or_none",
    "cryst_to_box_vectors",
    "assign_stereochemistry_from_3d",
    "__UNSET__",
    "dbg",
]

logger = logging.getLogger(__name__)


class __UNSET__:
    """Reference value for an unset parameter."""

    pass


def dbg[T](o: T, msg: str = "{}") -> T:
    if "{}" not in msg:
        msg += ": {}"
    logger.debug(msg.format(o))
    return o


def default_dict[K: Hashable, T, U](
    default_factory: Callable[[], T],
    map: Mapping[K, U] = {},
) -> DefaultDict[K, T | U]:
    dd: DefaultDict[K, T | U] = defaultdict(default_factory)
    dd.update(map)
    return dd


def no_none_in_values[K: Hashable, V](
    d: Mapping[K, V | None],
) -> TypeGuard[Mapping[K, V]]:
    return None not in d.values()


def option_to_iter[T](option: T | None) -> Iterator[T]:
    if option is not None:
        yield option


def unwrap[T](container: Iterable[T], msg: str = "") -> T:
    """
    Unwrap an iterable only if it has a single element; raise ValueError otherwise
    """
    if msg:
        msg += ": "

    iterator = iter(container)

    try:
        value = next(iterator)
    except StopIteration:
        raise ValueError(msg + "container has no elements")

    try:
        next(iterator)
    except StopIteration:
        return value

    raise ValueError(msg + "container has multiple elements")


def unwrap_or_none[T](container: Iterable[T]) -> T | None:
    """
    Unwrap an iterable only if it has a single element; return None otherwise
    """
    try:
        return unwrap(container)
    except ValueError:
        return None


def sort_tuple[*Ts](tup: tuple[*Ts]) -> tuple[*Ts]:
    return tuple(sorted(tup))  # pyright: ignore[reportArgumentType, reportUnknownVariableType, reportUnknownArgumentType]


def flatten[T](container: Iterable[Iterable[T]]) -> Iterator[T]:
    for inner in container:
        yield from inner


def try_or[**P, T, U](
    default: U,
    func: Callable[P, T],
    catch: type[BaseException],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> T | U:
    try:
        return func(*args, **kwargs)
    except catch:
        return default


def try_or_none[**P, T](
    func: Callable[P, T],
    catch: type[BaseException],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> T | None:
    return try_or(None, func, catch, *args, **kwargs)


def coerce_or_none[T](value: Any, func: Callable[[Any], T]) -> T | None:
    try:
        return func(value)
    except Exception:
        return None


def coerce_or_leave[T, U](value: T, func: Callable[[Any], U]) -> T | U:
    try:
        return func(value)
    except Exception:
        return value


def with_neighbours[T, U](
    iterable: Iterable[T],
    default: U = None,
) -> Iterator[tuple[T | U, T, T | U]]:
    """Return each element of the iterable with its neighbours.

        abcd -> _ab, abc, bcd, cd_

    The middle element of the tuple is the current element. Missing neighbours
    are set to ``default``. The resulting sequence has the same length as the
    original iterable.
    """
    iterator = iter(iterable)

    pred: T | U = default
    this: T
    succ: T | U

    try:
        this = next(iterator)
    except StopIteration:
        return

    for succ in iterator:
        yield (pred, this, succ)
        pred = this
        this = succ

    succ = default
    yield (pred, this, succ)


def float_or_unknown(s: str) -> float | None:
    if s == "?":
        return None
    return float(s)


def dec_hex(s: str) -> int:
    """
    Interpret a string as a decimal or hexadecimal integer.

    For a string of length n, the string is interpreted as decimal if the value
    is < 10^n. This makes the dec_hex representation identical to a decimal
    integer, except for strings that cannot be parsed as a decimal. For these
    strings, the first hexadecimal number with a leading digit greater than 9 is
    interpreted as 10^n, and subsequent numbers continue from there. For
    example, in PDB files, a fixed width column format, residue numbers for
    large systems sometimes follow thisrepresentation:

        "   1" -> 1
        "   2" -> 2
        ...
        "9999" -> 9999
        "A000" -> 10000
        "A001" -> 10001
        ...
        "A009" -> 10009
        "A00A" -> 10010
        "A00B" -> 10011
        ...
        "A00F" -> 10015
        "A010" -> 10016
        ...
        "FFFF" -> 34575

    Strings that can be interpreted as hex but do not have a leading hex digit
    greater than 9 are a value error, as are strings that cannot be interpreted
    as either decimal or hexadecimal integers:

        "10A2" -> ValueError
        " 2.3" -> ValueError
        "hiya" -> ValueError
    """

    try:
        return int(s, 10)
    except ValueError:
        n = len(s)
        parsed_as_hex = int(s, 16)
        smallest_hex: int = 0xA * 16 ** (n - 1)
        largest_dec: int = 10**n - 1
        if parsed_as_hex < smallest_hex:
            raise PabloError(f"hex value's leftmost character must be A-F: {s!r}")
        return parsed_as_hex - smallest_hex + largest_dec + 1


@overload
def charge_int_or_none(s: str, strict: Literal[True]) -> int: ...


@overload
def charge_int_or_none(s: str, strict: Literal[False]) -> int | None: ...


def charge_int_or_none(s: str, strict: bool = False):
    if s == "" or s == "?":
        return 0 if strict else None
    else:
        if s.strip() == "+" and not strict:
            return 1
        elif s.strip() == "-" and not strict:
            return -1
        elif s.endswith("+"):
            return int(s[:-1])
        elif s.endswith("-"):
            return -int(s[:-1])
        else:
            return int(s)


def cryst_to_box_vectors(  # pyright: ignore[reportUnknownParameterType]
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> Quantity:
    @no_type_check
    def inner(a, b, c, alpha, beta, gamma):
        import openmm.unit
        from openmm.app.internal.unitcell import computePeriodicBoxVectors
        from openmm.unit import nanometer as openmm_unit_nanometer

        box_vectors = computePeriodicBoxVectors(
            openmm.unit.Quantity(a, openmm.unit.angstrom),
            openmm.unit.Quantity(b, openmm.unit.angstrom),
            openmm.unit.Quantity(c, openmm.unit.angstrom),
            openmm.unit.Quantity(alpha, openmm.unit.degree),
            openmm.unit.Quantity(beta, openmm.unit.degree),
            openmm.unit.Quantity(gamma, openmm.unit.degree),
        )
        return box_vectors.value_in_unit(openmm_unit_nanometer) * unit.nanometer

    return inner(a, b, c, alpha, beta, gamma)  # pyright: ignore[reportUnknownVariableType]


def assign_stereochemistry_from_3d(molecule: MoleculeLike):
    @no_type_check
    def inner(molecule):
        from rdkit.Chem import AssignStereochemistryFrom3D, BondStereo

        if isinstance(molecule, _SimpleMolecule):
            # SimpleMolecules do not store stereo info
            return

        rdmol = molecule.to_rdkit()
        AssignStereochemistryFrom3D(rdmol, confId=0, replaceExistingTags=True)

        for offatom, rdatom in zip(molecule.atoms, rdmol.GetAtoms()):
            stereochemistry = None
            if rdatom.HasProp("_CIPCode"):
                stereo_code = rdatom.GetProp("_CIPCode")
                if stereo_code == "R":
                    stereochemistry = "R"
                elif stereo_code == "S":
                    stereochemistry = "S"
                else:
                    raise UndefinedStereochemistryError(
                        "In from_pdb: Expected atom stereochemistry of R or S. "
                        f"Got {stereo_code} instead.",
                    )
            offatom._stereochemistry = stereochemistry

        for offbond, rdbond in zip(molecule.bonds, rdmol.GetBonds()):
            stereochemistry = None
            tag = rdbond.GetStereo()
            if tag == BondStereo.STEREOZ:
                stereochemistry = "Z"
            elif tag == BondStereo.STEREOE:
                stereochemistry = "E"
            elif tag == BondStereo.STEREOTRANS or tag == BondStereo.STEREOCIS:
                raise PabloError(
                    f"Expected RDKit bond stereochemistry of E or Z, got {tag} instead",
                )
            offbond._stereochemistry = stereochemistry

    inner(molecule)


def asciify(s: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


Color = Iterable[float]
BondIndices = tuple[int, int]


def draw_molecule(
    molecule: Molecule | rdkit.Chem.rdchem.Mol,
    *,
    width: int = -1,
    height: int = 300,
    highlight_atoms: list[int] | dict[int, Color] | None = None,
    highlight_bonds: None | (list[BondIndices] | dict[BondIndices, Color]) = None,
    atom_notes: dict[int, str] | None = None,
    bond_notes: dict[BondIndices, str] | None = None,
    deemphasize_atoms: list[int] | None = None,
    explicit_hydrogens: bool | None = None,
    color_by_element: bool | None = None,
    legend: str = "",
) -> str:
    """Draw a molecule

    Returns a string encoding an XML with the requested drawing.

    Parameters
    ==========

    molecule
        The molecule to draw.
    image_width
        The width of the resulting image in pixels.
    image_height
        The height of the resulting image in pixels.
    highlight_atoms
        A list of atom indices to highlight, or a map from indices to colors.
        Colors should be given as triplets of floats between 0.0 and 1.0.
    highlight_bonds
        A list of pairs of atom indices indicating bonds to highlight, or a map
        from index pairs to colors. Colors should be given as triplets of floats
        between 0.0 and 1.0.
    atom_notes
        A map from atom indices to a string that should be printed near the
        atom.
    bond_notes
        A map from atom index pairs to a string that should be printed near the
        bond.
    deemphasize_atoms
        A list of atom indices to de-emphasize by drawing them and their bonds
        in light grey. Note that this changes the appearance of highlighted
        atoms and bonds.
    explicit_hydrogens
        If ``False``, allow uncharged monovalent hydrogens to be hidden. If
        ``True``, make all hydrogens explicit. If ``None``, defer to the
        provided molecule.
    color_by_element
        If True, color heteroatoms according to their element; if False, color
        atoms and bonds monochromatically. By default, uses black and white when
        highlight_atoms or highlight_bonds is provided, and color otherwise.

    Raises
    ======

    KeyError
        When an atom or bond in highlight_atoms or highlight_bonds is missing
        from the image, including when it is present in the molecule but hidden.

    Examples
    ========

    >>> from IPython.display import SVG
    >>>
    >>> SVG(draw_molecule(...))

    """

    # We're working in RDKit
    try:
        rdmol = rdkit.Chem.rdchem.Mol(molecule.to_rdkit())
    except AttributeError:
        rdmol = rdkit.Chem.rdchem.Mol(molecule)

    # Process color_by_element argument
    if color_by_element is None:
        color_by_element = highlight_atoms is None and highlight_bonds is None

    if color_by_element:
        set_atom_palette = lambda draw_options: draw_options.useDefaultAtomPalette()
    else:
        set_atom_palette = lambda draw_options: draw_options.useBWAtomPalette()

    # Process explicit_hydrogens argument
    # If we need to remove atoms, create a map from the original indices to the
    # new ones.
    if explicit_hydrogens is None:
        idx_map = {i: i for i in range(rdmol.GetNumAtoms())}
    elif explicit_hydrogens:
        idx_map = {i: i for i in range(rdmol.GetNumAtoms())}
        rdmol = rdkit.Chem.AddHs(rdmol, explicitOnly=True)
    else:
        idx_map = {
            old: new
            for new, old in enumerate(
                a.GetIdx()
                for a in rdmol.GetAtoms()
                if a.GetAtomicNum() != 1 and a.GetMass() != 1
            )
        }
        rdmol = rdkit.Chem.RemoveHs(rdmol, updateExplicitCount=True)

    # Process highlight_atoms argument for highlightAtoms and highlightAtomColors
    # highlightAtoms takes a list of atom indices
    # highlightAtomColors takes a mapping from atom indices to colors
    if highlight_atoms is None:
        highlight_atoms = []
        highlight_atom_colors = None
    elif isinstance(highlight_atoms, dict):
        highlight_atom_colors = {
            idx_map[i]: tuple(c) for i, c in highlight_atoms.items() if i in idx_map
        }
        highlight_atoms = list(highlight_atoms.keys())
    else:
        highlight_atoms = [idx_map[i] for i in highlight_atoms if i in idx_map]
        highlight_atom_colors = None

    # Process highlight_bonds argument for highlightBonds and highlightBondColors
    # highlightBonds takes a list of bond indices
    highlight_bond_indices: list[int]
    # highlightBondColors takes a mapping from bond indices to colors
    highlight_bond_colors: dict[int, Color] | None
    if highlight_bonds is None:
        highlight_bond_indices = []
        highlight_bond_colors = None
    elif isinstance(highlight_bonds, dict):
        highlight_bond_colors = {
            rdmol.GetBondBetweenAtoms(idx_map[i_a], idx_map[i_b]).GetIdx(): tuple(v)
            for (i_a, i_b), v in highlight_bonds.items()
            if i_a in idx_map and i_b in idx_map
        }

        highlight_bond_indices = list(highlight_bond_colors.keys())
    else:
        highlight_bond_indices = [
            rdmol.GetBondBetweenAtoms(idx_map[i_a], idx_map[i_b]).GetIdx()
            for i_a, i_b in highlight_bonds
            if i_a in idx_map and i_b in idx_map
        ]
        highlight_bond_colors = None

    # Process bond_notes argument and place notes in the molecule
    if bond_notes is not None:
        for (i_a, i_b), note in bond_notes.items():
            if i_a not in idx_map or i_b not in idx_map:
                continue
            rdbond = rdmol.GetBondBetweenAtoms(idx_map[i_a], idx_map[i_b])
            rdbond.SetProp("bondNote", asciify(str(note)))

    # Process atom_notes argument and place notes in the molecule
    if atom_notes is not None:
        for i, note in atom_notes.items():
            if i not in idx_map:
                continue
            rdatom = rdmol.GetAtomWithIdx(idx_map[i])
            rdatom.SetProp("atomNote", asciify(str(note)))

    # Resolve kekulization so it is the same for all drawn molecules
    rdkit.Chem.rdmolops.Kekulize(rdmol, clearAromaticFlags=True)

    # Compute 2D coordinates
    rdkit.Chem.rdDepictor.Compute2DCoords(rdmol)
    rdkit.Chem.rdDepictor.StraightenDepiction(rdmol)

    # Construct the drawing object and get a handle to its options
    drawer = rdkit.Chem.Draw.MolDraw2DSVG(width, height)
    draw_options = drawer.drawOptions()
    set_atom_palette(draw_options)
    draw_options.setBondNoteColour((0.7, 0.7, 0.7))

    # Set the colors used for each element according to the emphasize_atoms and
    # color_by_element arguments
    if deemphasize_atoms:
        draw_options.setHighlightColour((255 / 255, 176 / 255, 103 / 255))
        draw_options.continuousHighlight = False
        draw_options.circleAtoms = False
        highlight_atoms += deemphasize_atoms
        highlight_atom_colors = (
            {} if highlight_atom_colors is None else highlight_atom_colors
        )
        highlight_atom_colors.update({i: (0.8, 0.8, 0.8) for i in deemphasize_atoms})
    else:
        draw_options.setHighlightColour((255 / 255, 202 / 255, 154 / 255))

    # Draw the molecule
    # Note that if emphasize_atoms is used, this will be the un-emphasized parts
    # of the molecule
    drawer.DrawMolecule(
        rdmol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_atom_colors,
        highlightBonds=highlight_bond_indices,
        highlightBondColors=highlight_bond_colors,
        legend=legend,
    )

    # Finalize the SVG
    drawer.FinishDrawing()

    # Return an SVG object that we can view in notebook
    svg_contents = drawer.GetDrawingText()
    return svg_contents


def react(
    reactants: Sequence[Molecule],
    reaction_smarts: str,
) -> Iterable[tuple[Molecule, ...]]:
    # Convert reactants to rdmol, storing metadata as properties
    # Need to preserve metadata so we can identify leaving atoms and synonyms
    reactant_rdmols = [reactant.to_rdkit() for reactant in reactants]
    for reactant_rdmol, reactant_offmol in zip(reactant_rdmols, reactants):
        for reactant_rdatom, reactant_offatom in zip(
            reactant_rdmol.GetAtoms(),
            reactant_offmol.atoms,
        ):
            for key, value in reactant_offatom.metadata.items():
                if isinstance(value, bool):
                    reactant_rdatom.SetBoolProp(key, value)
                elif isinstance(value, int):
                    reactant_rdatom.SetIntProp(key, value)
                elif isinstance(value, float):
                    reactant_rdatom.SetDoubleProp(key, value)
                else:
                    reactant_rdatom.SetProp(key, str(value))

    # Prepare the reaction
    rxn = ReactionFromSmarts(reaction_smarts)
    product_rdmols = rxn.RunReactants(reactant_rdmols)

    # Get map from reaction SMARTS atom mappings to the equivalent OFF atom
    map_to_offatom: dict[Any, Atom] = {}
    for reactant_rdmol, reactant_offmol in zip(reactant_rdmols, reactants):
        assert rxn.IsMoleculeReactant(reactant_rdmol)
        for reactant_template in rxn.GetReactants():
            map_to_offatom.update(
                {
                    reactant_template.GetAtomWithIdx(query).GetProp(
                        "molAtomMapNumber",
                    ): reactant_offmol.atom(match)
                    for query, match in enumerate(
                        reactant_rdmol.GetSubstructMatch(reactant_template),
                    )
                    if reactant_template.GetAtomWithIdx(query).HasProp(
                        "molAtomMapNumber",
                    )
                },
            )

    # Process and yield the products
    for products in product_rdmols:
        # Skip products that cannot be sanitized
        try:
            for product in products:
                product.UpdatePropertyCache()
                rdkit.Chem.SanitizeMol(product)
        except rdkit.Chem.rdchem.MolSanitizeException:
            continue

        product_offmols = [Molecule.from_rdkit(product) for product in products]

        # Fix metadata of products
        for product_template in rxn.GetProducts():
            for product_rdmol, product_offmol in zip(products, product_offmols):
                # Go over atoms changed in the reaction and fix their metadata (rdkit often loses it)
                for product_idx, _product_template_idx in enumerate(
                    product_rdmol.GetSubstructMatch(product_template),
                ):
                    product_rdatom = product_rdmol.GetAtomWithIdx(product_idx)
                    product_offatom = product_offmol.atom(product_idx)

                    if product_rdatom.HasProp("old_mapno"):
                        rxn_map = product_rdatom.GetProp("old_mapno")
                        reactant_offatom = map_to_offatom[rxn_map]
                        product_offatom.metadata.update(reactant_offatom.metadata)
                        if "leaving_atom" in product_offatom.metadata:
                            product_offatom.metadata["leaving_atom"] = False
                        if "substructure_atom" in product_offatom.metadata:
                            product_offatom.metadata["substructure_atom"] = True
                        product_offatom.name = reactant_offatom.name

                # Copy the props back to the metadata
                for product_rdatom, product_offatom in zip(
                    product_rdmol.GetAtoms(),
                    product_offmol.atoms,
                ):
                    for key, value in product_rdatom.GetPropsAsDict().items():
                        try:
                            product_offatom.metadata[key] = value
                        except InvalidAtomMetadataError:
                            pass

        yield tuple(product_offmols)
