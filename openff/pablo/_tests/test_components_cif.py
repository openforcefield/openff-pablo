from pathlib import Path

import pytest
import xdg.BaseDirectory as xdg_base_dir
from openff.toolkit import Topology

from openff.pablo._pdb import topology_from_pdb
from openff.pablo._tests.utils import get_test_data_path, topology_identical_to_jsontop
from openff.pablo.ccd import CcdCache
from openff.pablo.ccd.patches import DEFAULT_PATCHES

COMPONENTS_CIF_PATH = Path(
    xdg_base_dir.xdg_cache_home,
    "openff-pablo/components.cif",
)


@pytest.mark.skipif(
    not COMPONENTS_CIF_PATH.is_file(),
)
@pytest.mark.slow
def test_components_cif():
    components_cache = CcdCache(
        library_paths=[COMPONENTS_CIF_PATH],
        fail_on_error=True,
    )

    components_cache["ALA"]


@pytest.mark.skipif(
    not COMPONENTS_CIF_PATH.is_file(),
)
@pytest.mark.slow
def test_components_cif_patched():
    components_cache = CcdCache(
        library_paths=[COMPONENTS_CIF_PATH],
        fail_on_error=True,
        patches=DEFAULT_PATCHES,
    )

    assert len(components_cache["ALA"]) > 1

    jsontopfile = "1A4T.json"
    pdbfile = "1A4T.pdb"

    jsontop_top = Topology.from_json(get_test_data_path(jsontopfile).read_text())
    pablo_top = topology_from_pdb(
        get_test_data_path(pdbfile),
        residue_library=components_cache,
    )
    topology_identical_to_jsontop(jsontop_top, pablo_top)
