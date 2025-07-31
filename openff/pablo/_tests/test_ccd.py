import pytest

from openff.pablo.ccd import CCD_RESIDUE_DEFINITION_CACHE
from openff.pablo.residue import ResidueDefinition


@pytest.mark.parametrize(
    "resname",
    [
        "EST",
    ],
)
def test_ccdcache_can_load_residues_with_internet(resname: str):
    CCD_RESIDUE_DEFINITION_CACHE[resname]


@pytest.mark.disable_socket
@pytest.mark.parametrize(
    "resname",
    [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "DG",
        "DA",
        "DT",
        "DC",
        "G",
        "A",
        "U",
        "C",
        "ACE",
        "NME",
        "NA",
        "CL",
        "BR",
        "CS",
        "IOD",
        "LI",
        "RB",
        "XE",
        "F",
        "K",
        "HOH",
    ],
)
def test_ccdcache_can_load_common_residues_without_internet(resname: str):
    CCD_RESIDUE_DEFINITION_CACHE[resname]


@pytest.mark.parametrize(
    ("resname", "expected_message"),
    [
        ("01", "unknown and absent from CCD"),
        ("UNK", "reserved residue name"),
        ("UNL", "reserved residue name"),
    ],
)
def test_ccdcache_gives_clear_errors(resname: str, expected_message: str):
    with pytest.raises(
        KeyError,
        match=expected_message,
        check=lambda e: e.args == (resname, expected_message),
    ):
        CCD_RESIDUE_DEFINITION_CACHE[resname]


def test_default_ccdcache_cys_definitions_unique():
    cys_defs = CCD_RESIDUE_DEFINITION_CACHE["CYS"]
    assert len(set(cys_defs)) == len(cys_defs)


@pytest.mark.slow
def test_ccdcache_with(hooh_def: ResidueDefinition, hoh_def: ResidueDefinition):
    assert "HOOH" not in CCD_RESIDUE_DEFINITION_CACHE
    new_ccdcache = CCD_RESIDUE_DEFINITION_CACHE.with_([hooh_def])
    assert new_ccdcache is not CCD_RESIDUE_DEFINITION_CACHE
    assert "HOOH" in new_ccdcache
    assert "HOOH" not in CCD_RESIDUE_DEFINITION_CACHE
    assert new_ccdcache["HOOH"] == [hooh_def]

    assert "HOH" in CCD_RESIDUE_DEFINITION_CACHE
    assert CCD_RESIDUE_DEFINITION_CACHE["HOH"] != hoh_def
    new_ccdcache = CCD_RESIDUE_DEFINITION_CACHE.with_({"HOH": [hoh_def]})
    assert new_ccdcache is not CCD_RESIDUE_DEFINITION_CACHE
    assert new_ccdcache["HOH"] == [*CCD_RESIDUE_DEFINITION_CACHE["HOH"], hoh_def]


def test_ccdcache_with_replaced(hoh_def: ResidueDefinition):
    assert "HOH" in CCD_RESIDUE_DEFINITION_CACHE
    assert CCD_RESIDUE_DEFINITION_CACHE["HOH"] != hoh_def
    new_ccdcache = CCD_RESIDUE_DEFINITION_CACHE.with_replaced({"HOH": [hoh_def]})
    assert new_ccdcache is not CCD_RESIDUE_DEFINITION_CACHE
    assert new_ccdcache["HOH"] == [hoh_def]


@pytest.mark.slow
def test_ccdcache_without(hooh_def: ResidueDefinition):
    assert "HOOH" not in CCD_RESIDUE_DEFINITION_CACHE
    ccdcache_with_hooh = CCD_RESIDUE_DEFINITION_CACHE.with_([hooh_def])
    new_ccdcache = ccdcache_with_hooh.without({"HOOH"})
    assert new_ccdcache is not ccdcache_with_hooh
    assert "HOOH" not in new_ccdcache
    assert "HOOH" in ccdcache_with_hooh
