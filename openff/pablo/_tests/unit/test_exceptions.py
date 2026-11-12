import io
from collections.abc import Iterable, Sequence

import pytest

from openff.pablo import topology_from_pdb
from openff.pablo._exceptions import create_pdb_residue_match_error
from openff.pablo._matching import ResidueConectMismatch, ResidueMismatch
from openff.pablo._pdb_data import PdbData
from openff.pablo._tests.utils import get_test_data_path
from openff.pablo.ccd._ccdcache import CcdCache
from openff.pablo.exceptions import PdbResidueMatchError
from openff.pablo.residue import ResidueDefinition, _Linkages


def test_auto_download_off_raises_clear_error_5eil(tmp_ccd_cache: CcdCache):
    with pytest.raises(
        PdbResidueMatchError,
        match="\n".join(
            [
                "",
                "Some missing residues are likely to be in the CCD; if the CCD",
                "form is what you intend, you can download them automatically",
                "by setting `residue_library.auto_download = True` or manually",
                "with the get_from_ccd method.",
            ],
        ),
    ):
        topology_from_pdb(
            get_test_data_path("prepared_pdbs/5eil_fixed.pdb"),
            residue_library=tmp_ccd_cache,
        )


@pytest.fixture
def cys_residue_data(cys_def: ResidueDefinition) -> PdbData:
    n_atoms = len(cys_def.atoms)
    return PdbData(
        name=[atom.name for atom in cys_def.atoms],
        res_name=["CYS"] * n_atoms,
        line_no=list(range(1, n_atoms + 1)),
        chain_id=["A"] * n_atoms,
        res_seq=["1"] * n_atoms,
    )


def mismatch(
    residue_definition: ResidueDefinition,
    reason: str,
    linkages: Iterable[_Linkages] = (),
    mismatch_type: type[ResidueMismatch] = ResidueMismatch,
) -> ResidueMismatch:
    return mismatch_type(
        residue_definition=residue_definition,
        index_to_atomdef=dict(enumerate(residue_definition.atoms)),
        reason=reason,
        linkages=frozenset(linkages),
    )


def residue_report(data: PdbData, mismatches: Sequence[ResidueMismatch]) -> str:
    """The report for a single residue's mismatches, without the shared header"""
    error = create_pdb_residue_match_error(data, [list(mismatches)], residue_library={})
    lines = str(error).splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("  A:CYS#1 "))
    end = next(
        (i for i in range(start, len(lines)) if lines[i] == ""),
        len(lines),
    )
    return "\n".join(lines[start:end])


def test_mismatches_differing_only_in_linkages_are_collapsed(
    cys_def: ResidueDefinition,
    cys_residue_data: PdbData,
):
    combinations = cys_def._linkage_combinations

    report = residue_report(
        cys_residue_data,
        [
            mismatch(cys_def, "first reason", combinations[:1]),
            mismatch(cys_def, "first reason", combinations[1:2]),
            mismatch(cys_def, "second reason", combinations),
        ],
    )

    assert report == "\n".join(
        [
            "  A:CYS#1 (l1-14): No matching residue definitions:",
            "    ╰─ CYSTEINE     didn't match: first reason (···, ·×·)",
            "                    didn't match: second reason",
        ],
    )


def test_complement_of_single_linkages_is_last(
    cys_def: ResidueDefinition,
    cys_residue_data: PdbData,
):
    combinations = cys_def._linkage_combinations

    report = residue_report(
        cys_residue_data,
        [
            mismatch(cys_def, "first reason", combinations[2:]),
            mismatch(cys_def, "second reason", combinations[:1]),
            mismatch(cys_def, "third reason", combinations[1:2]),
        ],
    )

    assert report == "\n".join(
        [
            "  A:CYS#1 (l1-14): No matching residue definitions:",
            "    ╰─ CYSTEINE ··· didn't match: second reason",
            "                ·×· didn't match: third reason",
            "                    didn't match: first reason",
        ],
    )


def test_incomplete_complement_lists_linkages(
    cys_def: ResidueDefinition,
    cys_residue_data: PdbData,
):
    combinations = cys_def._linkage_combinations

    report = residue_report(
        cys_residue_data,
        [
            mismatch(cys_def, "first reason", combinations[:1]),
            mismatch(cys_def, "second reason", combinations[2:]),
        ],
    )

    assert report == "\n".join(
        [
            "  A:CYS#1 (l1-14): No matching residue definitions:",
            "    ╰─ CYSTEINE ··· didn't match: first reason",
            "                    didn't match: second reason (··→, ·×→, ←··, ←×·, ←·→, ←×→)",
        ],
    )


def test_definitions_with_identical_mismatches_are_folded(
    cys_def: ResidueDefinition,
    cys_residue_data: PdbData,
):
    combinations = cys_def._linkage_combinations
    variant = cys_def.replace(description="CYSTEINE VARIANT")
    unrelated = cys_def.replace(description="UNRELATED")

    report = residue_report(
        cys_residue_data,
        [
            mismatch(cys_def, "first reason", combinations[:1]),
            mismatch(cys_def, "second reason", combinations[1:]),
            mismatch(variant, "first reason", combinations[:1]),
            mismatch(variant, "second reason", combinations[1:]),
            mismatch(unrelated, "other reason"),
        ],
    )

    assert report == "\n".join(
        [
            "  A:CYS#1 (l1-14): No matching residue definitions:",
            "    ├─ CYSTEINE, CYSTEINE VARIANT",
            "    │            ··· didn't match: first reason",
            "    │                didn't match: second reason",
            "    ╰─ UNRELATED     didn't match: other reason",
        ],
    )


def test_mismatches_to_one_definition_are_reported_together(
    cys_def: ResidueDefinition,
    cys_residue_data: PdbData,
):
    combinations = cys_def._linkage_combinations
    variant = cys_def.replace(description="CYSTEINE VARIANT")

    report = residue_report(
        cys_residue_data,
        [
            mismatch(
                cys_def,
                "CONECT-based reason",
                combinations[:1],
                ResidueConectMismatch,
            ),
            mismatch(variant, "other reason"),
            mismatch(cys_def, "name-based reason", combinations[1:2]),
        ],
    )

    assert report == "\n".join(
        [
            "  A:CYS#1 (l1-14): No matching residue definitions:",
            "    ├─ CYSTEINE         ··· didn't match: CONECT-based reason",
            "    │                   ·×· didn't match: name-based reason",
            "    ╰─ CYSTEINE VARIANT     didn't match: other reason",
        ],
    )


def test_report_2b0f_crosslink_and_linking_bond_on_same_atom(
    ytf_carboxy_crosslink_library: CcdCache,
):
    # 2B0F with NMA after YTF, so YTF's C can't form both its crosslink and its
    # posterior bond, and CYS 146 has a carboxy crosslink variant of every
    # definition
    with open(get_test_data_path("2B0F_ytf_nma.pdb")) as file:
        with pytest.raises(PdbResidueMatchError) as excinfo:
            topology_from_pdb(file, residue_library=ytf_carboxy_crosslink_library)

    assert str(excinfo.value) == "\n".join(
        [
            "some residues could not be identified",
            "A topology cannot be created without chemical information for every",
            "atom and bond. The following residues present in the PDB file",
            "could not be identified from the provided residue library:",
            "  A:CYS#146 (l2282-2291): No matching residue definitions:",
            "    ├─ CYSTEINE, CYSTEINE CARBOXY CROSSLINK",
            "    │   ←×→ didn't match: crosslink expected but no matching crosslink partner could be found",
            "    │       didn't match: Missing leaving atoms (H2, HG, HXT, OXT) are not explained by these combinations of bonds to other residues",
            "    ├─ CYSTEINE -HXT, CYSTEINE -HXT CARBOXY CROSSLINK",
            "    │   ←×→ didn't match: crosslink expected but no matching crosslink partner could be found",
            "    │       didn't match: Missing leaving atoms (H2, HG, OXT) are not explained by these combinations of bonds to other residues",
            "    ├─ CYSTEINE -H2, CYSTEINE -H2 CARBOXY CROSSLINK",
            "    │   ·×→ didn't match: Bond to previous residue not expected but required by neighbor",
            "    │       didn't match: Missing leaving atoms (HG, HXT, OXT) are not explained by these combinations of bonds to other residues",
            "    ├─ CYSTEINE -HXT -H2, CYSTEINE -HXT -H2 CARBOXY CROSSLINK",
            "    │   ·×→ didn't match: Bond to previous residue not expected but required by neighbor",
            "    │       didn't match: Missing leaving atoms (HG, OXT) are not explained by these combinations of bonds to other residues",
            "    ├─ CYSTEINE -HG, CYSTEINE -HG CARBOXY CROSSLINK",
            "    │   ←·→ didn't match: found CONECT record that could not be matched with a bond",
            "    │       didn't match: Missing leaving atoms (H2, HXT, OXT) are not explained by these combinations of bonds to other residues",
            "    ├─ CYSTEINE -HXT -HG, CYSTEINE -HXT -HG CARBOXY CROSSLINK",
            "    │   ←·→ didn't match: found CONECT record that could not be matched with a bond",
            "    │       didn't match: Missing leaving atoms (H2, OXT) are not explained by these combinations of bonds to other residues",
            "    ├─ CYSTEINE -H2 -HG, CYSTEINE -H2 -HG CARBOXY CROSSLINK",
            "    │   ··→ didn't match: Bond to previous residue not expected but required by neighbor",
            "    │   ··· didn't match: Missing leaving atoms (HXT, OXT) are not explained by this combination of bonds to other residues",
            "    ├─ CYSTEINE -HXT -H2 -HG, CYSTEINE -HXT -H2 -HG CARBOXY CROSSLINK",
            "    │   ··→ didn't match: Bond to previous residue not expected but required by neighbor",
            "    │   ··· didn't match: Missing leaving atoms (OXT) are not explained by this combination of bonds to other residues",
            "    ╰─ CYSTEINE +H3, CYSTEINE -HXT +H3, CYSTEINE -H2 +H3, CYSTEINE -HXT -H2 +H3, CYSTEINE -HG +H3, CYSTEINE -HXT -HG +H3, CYSTEINE -H2 -HG +H3, CYSTEINE -HXT -H2 -HG +H3, CYSTEINE +H3 CARBOXY CROSSLINK, CYSTEINE -HXT +H3 CARBOXY CROSSLINK, CYSTEINE -H2 +H3 CARBOXY CROSSLINK, CYSTEINE -HXT -H2 +H3 CARBOXY CROSSLINK, CYSTEINE -HG +H3 CARBOXY CROSSLINK, CYSTEINE -HXT -HG +H3 CARBOXY CROSSLINK, CYSTEINE -H2 -HG +H3 CARBOXY CROSSLINK, CYSTEINE -HXT -H2 -HG +H3 CARBOXY CROSSLINK",
            "            didn't match: Expected atoms are missing: H3 (all present atoms were matched)",
            "",
            "  B:YTF#7 (l2926-2955): No matching residue definitions:",
            "    ╰─ ethyl (4~{S})-4,7-bis(azanyl)-7-oxidanylidene-heptanoate ←×· didn't match: Bond to next residue not expected but required by neighbor",
            "                                                                ←·→ didn't match: found CONECT record that could not be matched with a bond",
            "                                                                    didn't match: Missing leaving atoms (H11, H2) are not explained by these combinations of bonds to other residues",
            "",
            "Some missing residues are likely to be in the CCD; if the CCD",
            "form is what you intend, you can download them automatically",
            "by setting `residue_library.auto_download = True` or manually",
            "with the get_from_ccd method.",
        ],
    )


def test_report_vicinal_disulfide_with_partial_leaving_fragment(
    tmp_ccd_cache: CcdCache,
):
    # CYS 222 is missing OXT but not HXT, so no combination of linkages fits it
    pdb = "".join(
        line
        for line in get_test_data_path("3cu9_vicinal_disulfide.pdb")
        .read_text()
        .splitlines(keepends=True)
        if " OXT CYS A 222" not in line
    )

    with pytest.raises(PdbResidueMatchError) as excinfo:
        topology_from_pdb(io.StringIO(pdb), residue_library=tmp_ccd_cache)

    assert str(excinfo.value) == "\n".join(
        [
            "some residues could not be identified",
            "A topology cannot be created without chemical information for every",
            "atom and bond. The following residues present in the PDB file",
            "could not be identified from the provided residue library:",
            "  A:CYS#221 (l1-11): No matching residue definitions:",
            "    ├─ CYSTEINE                  ·×→ didn't match: Bond to next residue expected but not supported by neighbor",
            "    │                                didn't match: Missing leaving atoms (HG, HXT, OXT) are not explained by these combinations of bonds to other residues",
            "    ├─ CYSTEINE -HXT             ·×→ didn't match: Bond to next residue expected but not supported by neighbor",
            "    │                                didn't match: Missing leaving atoms (HG, OXT) are not explained by these combinations of bonds to other residues",
            "    ├─ CYSTEINE -H2                  didn't match: Unknown atom names: H2 (expected HG; OXT; HXT; different number of atoms were expected than were unmatched)",
            "    ├─ CYSTEINE -HXT -H2             didn't match: Unknown atom names: H2 (expected HG; OXT; different number of atoms were expected than were unmatched)",
            "    ├─ CYSTEINE -HG              ··→ didn't match: Bond to next residue expected but not supported by neighbor",
            "    │                                didn't match: Missing leaving atoms (HXT, OXT) are not explained by these combinations of bonds to other residues",
            "    ├─ CYSTEINE -HXT -HG         ··→ didn't match: Bond to next residue expected but not supported by neighbor",
            "    │                                didn't match: Missing leaving atoms (OXT) are not explained by these combinations of bonds to other residues",
            "    ├─ CYSTEINE -H2 -HG              didn't match: Unknown atom names: H2 (expected OXT; HXT; different number of atoms were expected than were unmatched)",
            "    ├─ CYSTEINE -HXT -H2 -HG         didn't match: Unknown atom names: H2 (expected OXT)",
            "    ├─ CYSTEINE +H3, CYSTEINE -HXT +H3, CYSTEINE -HG +H3, CYSTEINE -HXT -HG +H3",
            "    │                                didn't match: Expected atoms are missing: H3 (all present atoms were matched)",
            "    ├─ CYSTEINE -H2 +H3              didn't match: Unknown atom names: H2 (expected HG; H3; OXT; HXT; different number of atoms were expected than were unmatched)",
            "    ├─ CYSTEINE -HXT -H2 +H3         didn't match: Unknown atom names: H2 (expected HG; H3; OXT; different number of atoms were expected than were unmatched)",
            "    ├─ CYSTEINE -H2 -HG +H3          didn't match: Unknown atom names: H2 (expected H3; OXT; HXT; different number of atoms were expected than were unmatched)",
            "    ╰─ CYSTEINE -HXT -H2 -HG +H3     didn't match: Unknown atom names: H2 (expected H3; OXT; different number of atoms were expected than were unmatched)",
            "",
            "  A:CYS#222 (l12-22): No matching residue definitions:",
            "    ├─ CYSTEINE                      didn't match: Missing leaving atoms (H2, HG, OXT) are not explained by any combination of bonds to other residues",
            "    ├─ CYSTEINE -HXT                 didn't match: Unknown atom names: HXT (expected HG; OXT; H2 or synonyms HN2; different number of atoms were expected than were unmatched)",
            "    ├─ CYSTEINE -H2                  didn't match: Missing leaving atoms (HG, OXT) are not explained by any combination of bonds to other residues",
            "    ├─ CYSTEINE -HXT -H2             didn't match: Unknown atom names: HXT (expected HG; OXT; different number of atoms were expected than were unmatched)",
            "    ├─ CYSTEINE -HG                  didn't match: Missing leaving atoms (H2, OXT) are not explained by any combination of bonds to other residues",
            "    ├─ CYSTEINE -HXT -HG             didn't match: Unknown atom names: HXT (expected OXT; H2 or synonyms HN2; different number of atoms were expected than were unmatched)",
            "    ├─ CYSTEINE -H2 -HG              didn't match: Missing leaving atoms (OXT) are not explained by any combination of bonds to other residues",
            "    ├─ CYSTEINE -HXT -H2 -HG         didn't match: Unknown atom names: HXT (expected OXT)",
            "    ├─ CYSTEINE +H3, CYSTEINE -H2 +H3, CYSTEINE -HG +H3, CYSTEINE -H2 -HG +H3",
            "    │                                didn't match: Expected atoms are missing: H3 (all present atoms were matched)",
            "    ├─ CYSTEINE -HXT +H3             didn't match: Unknown atom names: HXT (expected HG; H3; OXT; H2 or synonyms HN2; different number of atoms were expected than were unmatched)",
            "    ├─ CYSTEINE -HXT -H2 +H3         didn't match: Unknown atom names: HXT (expected HG; H3; OXT; different number of atoms were expected than were unmatched)",
            "    ├─ CYSTEINE -HXT -HG +H3         didn't match: Unknown atom names: HXT (expected H3; OXT; H2 or synonyms HN2; different number of atoms were expected than were unmatched)",
            "    ╰─ CYSTEINE -HXT -H2 -HG +H3     didn't match: Unknown atom names: HXT (expected H3; OXT; different number of atoms were expected than were unmatched)",
            "",
            "Some missing residues are likely to be in the CCD; if the CCD",
            "form is what you intend, you can download them automatically",
            "by setting `residue_library.auto_download = True` or manually",
            "with the get_from_ccd method.",
        ],
    )
