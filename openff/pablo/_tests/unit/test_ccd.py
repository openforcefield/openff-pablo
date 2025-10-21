from tempfile import TemporaryDirectory

import pytest

from openff.pablo._utils import unwrap
from openff.pablo.ccd import STD_CCD_CACHE
from openff.pablo.ccd._ccdcache import CcdCache
from openff.pablo.residue import ResidueDefinition


@pytest.mark.parametrize(
    "resname",
    [
        "EST",
        "GDP",
    ],
)
def test_ccdcache_can_load_residues_with_internet(resname: str):
    STD_CCD_CACHE[resname]


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
    STD_CCD_CACHE[resname]


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
        STD_CCD_CACHE[resname]


def test_default_ccdcache_cys_definitions_unique():
    cys_defs = STD_CCD_CACHE["CYS"]
    assert len(set(cys_defs)) == len(cys_defs)


@pytest.mark.slow
def test_ccdcache_with(hooh_def: ResidueDefinition, hoh_def: ResidueDefinition):
    assert "HOOH" not in STD_CCD_CACHE
    new_ccdcache = STD_CCD_CACHE.with_([hooh_def])
    assert new_ccdcache is not STD_CCD_CACHE
    assert "HOOH" in new_ccdcache
    assert "HOOH" not in STD_CCD_CACHE
    assert new_ccdcache["HOOH"] == [hooh_def]

    assert "HOH" in STD_CCD_CACHE
    assert STD_CCD_CACHE["HOH"] != hoh_def
    new_ccdcache = STD_CCD_CACHE.with_({"HOH": [hoh_def]})
    assert new_ccdcache is not STD_CCD_CACHE
    assert new_ccdcache["HOH"] == [*STD_CCD_CACHE["HOH"], hoh_def]


def test_ccdcache_with_replaced(hoh_def: ResidueDefinition):
    assert "HOH" in STD_CCD_CACHE
    assert STD_CCD_CACHE["HOH"] != hoh_def
    new_ccdcache = STD_CCD_CACHE.with_replaced({"HOH": [hoh_def]})
    assert new_ccdcache is not STD_CCD_CACHE
    assert new_ccdcache["HOH"] == [hoh_def]


@pytest.mark.slow
def test_ccdcache_without(hooh_def: ResidueDefinition):
    assert "HOOH" not in STD_CCD_CACHE
    ccdcache_with_hooh = STD_CCD_CACHE.with_([hooh_def])
    new_ccdcache = ccdcache_with_hooh.without({"HOOH"})
    assert new_ccdcache is not ccdcache_with_hooh
    assert "HOOH" not in new_ccdcache
    assert "HOOH" in ccdcache_with_hooh


def test_with_patch():
    def test_patch(resdef: ResidueDefinition) -> list[ResidueDefinition]:
        return [
            resdef.replace(description=resdef.description + " test with_patch 1"),
            resdef.replace(description=resdef.description + " test with_patch 2"),
        ]

    with (
        TemporaryDirectory() as tmpdir0,
        TemporaryDirectory() as tmpdir1,
        TemporaryDirectory() as tmpdir2,
        TemporaryDirectory() as tmpdir3,
    ):
        cache0: CcdCache = CcdCache(
            library_paths=[],
            cache_path=tmpdir0,
            patches=[{"ACE": test_patch}],
            preload=["ACE"],
        )

        cache1: CcdCache = CcdCache(
            library_paths=[],
            cache_path=tmpdir1,
            patches=[{"ACE": test_patch}],
        )

        cache2: CcdCache = CcdCache(
            library_paths=[],
            cache_path=tmpdir2,
        ).with_patch("ACE", test_patch)

        cache3: CcdCache = CcdCache(
            library_paths=[],
            cache_path=tmpdir3,
            preload=["ACE"],
        ).with_patch("ACE", test_patch)

        correct_desc_seq = [
            "ACETYL GROUP test with_patch 1",
            "ACETYL GROUP test with_patch 2",
        ]
        assert [resdef.description for resdef in cache0["ACE"]] == correct_desc_seq, (
            "cache0 is out of order (patches arg with preload)"
        )
        assert [resdef.description for resdef in cache1["ACE"]] == correct_desc_seq, (
            "cache1 is out of order (patches arg without preload)"
        )
        assert [resdef.description for resdef in cache2["ACE"]] == correct_desc_seq, (
            "cache2 is out of order (with_patches method without preload)"
        )
        assert [resdef.description for resdef in cache3["ACE"]] == correct_desc_seq, (
            "cache3 is out of order (with_patches method with preload)"
        )

        assert cache0 == cache1
        assert cache1 == cache2
        assert cache2 == cache3
        assert cache0._patches == cache1._patches
        assert cache1._patches == cache2._patches
        assert cache2._patches == cache3._patches
        assert cache0._definitions == cache1._definitions
        assert cache1._definitions == cache2._definitions
        assert cache2._definitions == cache3._definitions
        assert cache0["ACE"] == cache1["ACE"]
        assert cache1["ACE"] == cache2["ACE"]
        assert cache2["ACE"] == cache3["ACE"]


def test_water():
    hoh = unwrap(STD_CCD_CACHE["HOH"])
    wat = unwrap(STD_CCD_CACHE["WAT"])
    assert hoh.to_openff_molecule().is_isomorphic_with(wat.to_openff_molecule())
    assert hoh.virtual_sites == ()
    assert wat.virtual_sites == ()
    assert [atom.name for atom in hoh.atoms] == ["O", "H1", "H2"]
    assert [atom.name for atom in wat.atoms] == ["O", "H1", "H2"]

    ccd_with_sol = STD_CCD_CACHE.with_([hoh.replace(residue_name="SOL")])

    sol = unwrap(
        resdef for resdef in ccd_with_sol["SOL"] if resdef.n_expected_atoms == 3
    )
    assert sol.virtual_sites == ()
    assert [atom.name for atom in sol.atoms] == ["O", "H1", "H2"]

    ccd_with_vsites = ccd_with_sol.with_vsite_water()

    hoh_vsites = ccd_with_vsites["HOH"]
    assert len(hoh_vsites) == 3
    assert hoh_vsites[0] == hoh
    assert hoh_vsites[1] == hoh.replace(virtual_sites=["EPW"])
    assert hoh_vsites[2] == hoh.replace(virtual_sites=["EP1", "EP2"])

    wat_vsites = ccd_with_vsites["WAT"]
    assert len(wat_vsites) == 3
    assert wat_vsites[0] == wat
    assert wat_vsites[1] == wat.replace(virtual_sites=["EPW"])
    assert wat_vsites[2] == wat.replace(virtual_sites=["EP1", "EP2"])

    sol_vsites = [resdef for resdef in ccd_with_vsites["SOL"] if len(resdef.atoms) == 3]
    assert len(sol_vsites) == 3
    assert sol_vsites[0] == sol
    assert sol_vsites[1] == sol.replace(virtual_sites=["EPW"])
    assert sol_vsites[2] == sol.replace(virtual_sites=["EP1", "EP2"])
