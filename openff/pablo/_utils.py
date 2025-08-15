import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import (
    Any,
    DefaultDict,
    Literal,
    ParamSpec,
    TypeGuard,
    TypeVar,
    TypeVarTuple,
    no_type_check,
    overload,
)

import rdkit
import rdkit.Chem
import rdkit.Chem.Draw
import rdkit.Chem.rdDepictor
import rdkit.Geometry
from openff.toolkit import Molecule
from openff.toolkit.topology._mm_molecule import _SimpleMolecule
from openff.toolkit.topology.molecule import MoleculeLike
from openff.toolkit.utils import UndefinedStereochemistryError
from openff.units import unit
from pint import Quantity

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

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
Ts = TypeVarTuple("Ts")
P = ParamSpec("P")


class __UNSET__:
    pass


def dbg(o: T, msg: str = "{}") -> T:
    if "{}" not in msg:
        msg += ": {}"
    logging.debug(msg.format(o))
    return o


def default_dict(
    default_factory: Callable[[], T],
    map: Mapping[U, V] = {},
) -> DefaultDict[U, T | V]:
    dd: DefaultDict[U, T | V] = defaultdict(default_factory)
    dd.update(map)
    return dd


def no_none_in_values(d: Mapping[T, U | None]) -> TypeGuard[Mapping[T, U]]:
    return None not in d.values()


def option_to_iter(option: T | None) -> Iterator[T]:
    if option is not None:
        yield option


def unwrap(container: Iterable[T], msg: str = "") -> T:
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


def unwrap_or_none(container: Iterable[T]) -> T | None:
    """
    Unwrap an iterable only if it has a single element; return None otherwise
    """
    try:
        return unwrap(container)
    except ValueError:
        return None


def sort_tuple(tup: tuple[*Ts]) -> tuple[*Ts]:
    return tuple(sorted(tup))  # type: ignore


def flatten(container: Iterable[Iterable[T]]) -> Iterator[T]:
    for inner in container:
        yield from inner


def try_or(
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


def try_or_none(
    func: Callable[P, T],
    catch: type[BaseException],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> T | None:
    return try_or(None, func, catch, *args, **kwargs)


def coerce_or_none(value: Any, func: Callable[[Any], T]) -> T | None:
    try:
        return func(value)
    except Exception:
        return None


def with_neighbours(
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
            raise ValueError(f"hex value's leftmost character must be A-F: {s!r}")
        return parsed_as_hex - smallest_hex + largest_dec + 1


@overload
def charge_int_or_none(s: str, strict: Literal[True]) -> int: ...


@overload
def charge_int_or_none(s: str, strict: Literal[False]) -> int | None: ...


def charge_int_or_none(s: str, strict: bool = False):
    if s == "" or s == "?":
        return 0 if strict else None
    else:
        if s.endswith("+"):
            return int(s[:-1])
        elif s.endswith("-"):
            return -int(s[:-1])
        else:
            return int(s)


def cryst_to_box_vectors(
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

    return inner(a, b, c, alpha, beta, gamma)  # type: ignore


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
                raise ValueError(
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
    width: int = 500,
    height: int = 500,
    highlight_atoms: list[int] | dict[int, Color] | None = None,
    highlight_bonds: None | (list[BondIndices] | dict[BondIndices, Color]) = None,
    atom_notes: dict[int, str] | None = None,
    bond_notes: dict[BondIndices, str] | None = None,
    emphasize_atoms: list[int] | None = None,
    explicit_hydrogens: bool | None = None,
    color_by_element: bool | None = None,
):
    """Draw a molecule

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
    emphasize_atoms
        A list of atom indices to emphasize by drawing other atoms (and their
        bonds) in light grey.
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
    """
    from IPython.display import SVG

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
                for a in rdmol.GetAtoms()  # type: ignore
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

    # Construct the drawing object and get a handle to its options
    drawer = rdkit.Chem.Draw.MolDraw2DSVG(width, height)
    draw_options = drawer.drawOptions()

    # Specify the scale to fit all atoms
    # This is important for emphasize_atoms
    coords_2d = next(iter(rdmol.GetConformers())).GetPositions()[..., (0, 1)]
    drawer.SetScale(
        width,
        height,
        rdkit.Geometry.rdGeometry.Point2D(*(coords_2d.min(axis=0) - 1.0)),
        rdkit.Geometry.rdGeometry.Point2D(*(coords_2d.max(axis=0) + 1.0)),
    )

    # Set the colors used for each element according to the emphasize_atoms and
    # color_by_element arguments
    if emphasize_atoms:
        draw_options.setAtomPalette(
            {i: (0.8, 0.8, 0.8) for i in range(rdmol.GetNumAtoms())},
        )
    else:
        set_atom_palette(draw_options)

    # Draw the molecule
    # Note that if emphasize_atoms is used, this will be the un-emphasized parts
    # of the molecule
    drawer.DrawMolecule(
        rdmol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_atom_colors,
        highlightBonds=highlight_bond_indices,
        highlightBondColors=highlight_bond_colors,
    )

    # Draw an overlapping molecule for the emphasized atoms
    if emphasize_atoms:
        # Set the atom palette according to the color_by_element argument
        set_atom_palette(draw_options)

        # Create a copy of the molecule that removes atoms that aren't emphasized
        emphasized = rdkit.Chem.rdchem.RWMol(rdmol)
        emphasized.BeginBatchEdit()
        for i in set(idx_map) - set(emphasize_atoms):
            emphasized.RemoveAtom(idx_map[i])
        emphasized.CommitBatchEdit()

        # Draw the molecule. The scale has been fixed and we're re-using the
        # same coordinates, so this will overlap the background molecule
        drawer.DrawMolecule(emphasized)

    # Finalize the SVG
    drawer.FinishDrawing()

    # Return an SVG object that we can view in notebook
    svg_contents = drawer.GetDrawingText()
    return SVG(svg_contents)
