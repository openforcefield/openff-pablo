import pytest

from openff.pablo._rdkit import RdMol


def test_split_simple_molecule_fragments():
    smiles = "[H][O][H].[H][O][H]"
    rdmol = RdMol.from_smiles(smiles)
    fragments = list(rdmol.split_molecule_fragments())
    assert len(fragments) == 2
    print([fragment.to_smiles() for fragment in fragments])
    a, b = fragments
    assert a.n_atoms == 3
    assert b.n_atoms == 3


@pytest.mark.slow
def test_split_many_molecule_fragments():
    n = 9999
    smiles = ".".join(["[H][O][H]"] * n)
    rdmol = RdMol.from_smiles(smiles)
    fragments = list(rdmol.split_molecule_fragments())
    assert len(fragments) == n
    assert all(fragment.n_atoms == 3 for fragment in fragments)


def test_empty_conformers():
    rdmol = RdMol.from_smiles("O")
    assert rdmol.conformers().shape == (0, 3, 3)
