import pytest

from openff.pablo import topology_from_pdb
from openff.pablo._tests.utils import get_test_data_path
from openff.pablo.ccd._ccdcache import CcdCache
from openff.pablo.exceptions import PdbResidueMatchError


def test_auto_download_off_raises_clear_error_5eil(tmp_ccd_cache: CcdCache):
    with pytest.raises(
        PdbResidueMatchError,
        match="\n".join(
            [
                "",
                "Some missing residues are likely to be in the CCD; you can download",
                "them automatically by setting `residue_library.auto_download = True`",
                "or manually with the get_from_ccd method.",
            ],
        ),
    ):
        topology_from_pdb(
            get_test_data_path("prepared_pdbs/5eil_fixed.pdb"),
            residue_library=tmp_ccd_cache,
        )
