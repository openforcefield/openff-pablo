from collections.abc import Iterable

import pytest

from openff.pablo import topology_from_pdb
from openff.pablo._exceptions import create_pdb_residue_match_error
from openff.pablo._matching import ResidueMismatch
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


def cys_mismatch(
    cys_def: ResidueDefinition,
    reason: str,
    linkages: Iterable[_Linkages],
) -> ResidueMismatch:
    return ResidueMismatch(
        residue_definition=cys_def,
        index_to_atomdef=dict(enumerate(cys_def.atoms)),
        reason=reason,
        linkages=frozenset(linkages),
    )


def mismatch_lines(error: PdbResidueMatchError) -> list[str]:
    return [line for line in str(error).splitlines() if "didn't match" in line]


def test_mismatches_differing_only_in_linkages_are_collapsed(
    cys_def: ResidueDefinition,
    cys_residue_data: PdbData,
):
    combinations = cys_def._linkage_combinations

    error = create_pdb_residue_match_error(
        cys_residue_data,
        [
            [
                cys_mismatch(cys_def, "first reason", combinations[:1]),
                cys_mismatch(cys_def, "first reason", combinations[1:2]),
                cys_mismatch(cys_def, "second reason", combinations),
            ],
        ],
        residue_library={},
    )

    lines = mismatch_lines(error)
    assert len(lines) == 2
    assert lines[0].endswith(
        "didn't match: first reason"
        + f" ({combinations[0].report()}, {combinations[1].report()})",
    )
    assert lines[1].endswith("didn't match: second reason (all linkages)")


def test_complement_of_single_linkages_is_all_other_linkages(
    cys_def: ResidueDefinition,
    cys_residue_data: PdbData,
):
    combinations = cys_def._linkage_combinations

    error = create_pdb_residue_match_error(
        cys_residue_data,
        [
            [
                cys_mismatch(cys_def, "first reason", combinations[2:]),
                cys_mismatch(cys_def, "second reason", combinations[:1]),
                cys_mismatch(cys_def, "third reason", combinations[1:2]),
            ],
        ],
        residue_library={},
    )

    lines = mismatch_lines(error)
    assert len(lines) == 3
    assert "second reason" in lines[0]
    assert "third reason" in lines[1]
    assert lines[2].endswith("didn't match: first reason (all other linkages)")


def test_incomplete_complement_lists_linkages(
    cys_def: ResidueDefinition,
    cys_residue_data: PdbData,
):
    combinations = cys_def._linkage_combinations

    error = create_pdb_residue_match_error(
        cys_residue_data,
        [
            [
                cys_mismatch(cys_def, "first reason", combinations[:1]),
                cys_mismatch(cys_def, "second reason", combinations[2:]),
            ],
        ],
        residue_library={},
    )

    lines = mismatch_lines(error)
    reports = ", ".join(combination.report() for combination in combinations[2:])
    assert lines[1].endswith(f"didn't match: second reason ({reports})")
