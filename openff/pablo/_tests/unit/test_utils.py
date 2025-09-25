import pytest
from openff.toolkit import Molecule

from openff.pablo._utils import (
    __UNSET__,
    assign_stereochemistry_from_3d,
    dec_hex,
    flatten,
    sort_tuple,
    unwrap,
    with_neighbours,
)


def test_unset_is_self():
    assert __UNSET__ is __UNSET__


def test_unset_is_not_none():
    assert __UNSET__ is not None


def test_unwrap_raises_on_multiple_elements():
    with pytest.raises(ValueError):
        unwrap([1, 2, 3])


def test_unwrap_raises_on_zero_elements():
    with pytest.raises(ValueError):
        unwrap([])


def test_unwrap_unwraps_on_one_element():
    assert unwrap([1]) == 1


def test_sort_tuple():
    unsorted = (0, 2, 1)
    sorted = (0, 1, 2)
    assert sort_tuple(unsorted) == sorted
    assert isinstance(sort_tuple(unsorted), tuple)


def test_flatten():
    assert list(flatten(["asd", "fdg"])) == list("asdfdg")


@pytest.mark.parametrize("iterable", ["abcd", "a", ""])
def test_with_neighbours(iterable: str):
    processed = list(with_neighbours(iterable))
    assert len(processed) == len(iterable)
    assert [e for _b, e, _a in processed] == list(iterable)
    assert [b for b, _e, _a in processed][1:] == list(iterable)[:-1]
    assert [a for _b, _e, a in processed][:-1] == list(iterable)[1:]
    if len(iterable) != 0:
        assert processed[0][0] is None
        assert processed[-1][-1] is None


@pytest.mark.parametrize(
    ("s", "n"),
    [
        ("   1", 1),
        ("   2", 2),
        ("9999", 9999),
        ("A000", 10000),
        ("A001", 10001),
        ("A009", 10009),
        ("A00A", 10010),
        ("A00B", 10011),
        ("A00F", 10015),
        ("A010", 10016),
        ("FFFF", 34575),
        ("99999", 99999),
        ("A0000", 100000),
        ("FFFFF", 493215),
    ],
)
def test_dec_hex(s: str, n: int):
    assert dec_hex(s) == n


@pytest.mark.parametrize(
    "s",
    [
        ("10A2"),
        (" 2.3"),
        ("hiya"),
    ],
)
def test_dec_hex_raises(
    s: str,
):
    with pytest.raises(ValueError):
        dec_hex(s)


def test_assign_stereochemistry_from_3d_bonds():
    cis_smiles = r"[Cl:1]\[C:2]([H:6])=[C:3]([H:5])/[Cl:4]"
    trans_smiles = r"[Cl:1]\[C:2]([H:6])=[C:3]([H:5])\[Cl:4]"
    undef_smiles = r"[Cl:1][C:2]([H:6])=[C:3]([H:5])[Cl:4]"

    cis_mol = Molecule.from_mapped_smiles(cis_smiles)
    trans_mol = Molecule.from_mapped_smiles(trans_smiles)
    undef_mol = Molecule.from_mapped_smiles(undef_smiles, allow_undefined_stereo=True)

    cis_mol.generate_conformers(n_conformers=1)
    trans_mol.generate_conformers(n_conformers=1)

    undef_mol.add_conformer(cis_mol.conformers[0])  # type:ignore
    assign_stereochemistry_from_3d(undef_mol)
    assert "Z" in [bond.stereochemistry for bond in undef_mol.bonds]

    undef_mol.generate_conformers(n_conformers=0)

    undef_mol.add_conformer(trans_mol.conformers[0])  # type:ignore
    assign_stereochemistry_from_3d(undef_mol)
    assert "E" in [bond.stereochemistry for bond in undef_mol.bonds]


def test_assign_stereochemistry_from_3d_atoms():
    r_smiles = r"[Cl:2][C@:1]([F:3])([H:5])[O-:4]"
    s_smiles = r"[Cl:2][C@@:1]([F:3])([H:5])[O-:4]"
    undef_smiles = r"[Cl:2][C:1]([F:3])([H:5])[O-:4]"

    r_mol = Molecule.from_mapped_smiles(r_smiles)
    s_mol = Molecule.from_mapped_smiles(s_smiles)
    undef_mol = Molecule.from_mapped_smiles(undef_smiles, allow_undefined_stereo=True)

    r_mol.generate_conformers(n_conformers=1)
    s_mol.generate_conformers(n_conformers=1)

    assert r_mol.conformers is not None
    assert s_mol.conformers is not None

    undef_mol.add_conformer(r_mol.conformers[0])
    assign_stereochemistry_from_3d(undef_mol)
    assert undef_mol.atom(0).stereochemistry == "R"

    undef_mol.generate_conformers(n_conformers=0)

    undef_mol.add_conformer(s_mol.conformers[0])
    assign_stereochemistry_from_3d(undef_mol)
    assert undef_mol.atom(0).stereochemistry == "S"
