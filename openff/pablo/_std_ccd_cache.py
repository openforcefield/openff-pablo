from pathlib import Path

import xdg.BaseDirectory as xdg_base_dir

from openff.pablo.chem import PEPTIDE_BOND
from openff.pablo.residue import ResidueDefinition

from .ccd._ccdcache import CcdCache
from .ccd.patches import DEFAULT_PATCHES

__all__ = [
    "STD_CCD_CACHE",
]


# This is in a function so it can be precisely reproduced with a temporary cache
# directory for testing
def _construct_std_ccd_cache(
    cache_path: Path = Path(
        xdg_base_dir.save_cache_path("openff-pablo"),
        "ccd_cache",
    ),
) -> CcdCache:
    return CcdCache(
        # TODO: Use a proper resource setup for this
        library_paths=[Path(__file__).parent / "ccd/data/ccd_cache"],
        cache_path=cache_path,
        patches=DEFAULT_PATCHES,
        extra_definitions={
            "I": [ResidueDefinition.from_smiles("[I-:1]", {1: "I"}, "I")],
            "WAT": [
                ResidueDefinition.from_smiles(
                    "[H:2][O:1][H:3]",
                    {1: "O", 2: "H1", 3: "H2"},
                    "WAT",
                ).with_synonyms({"H1": ["1H"], "H2": ["2H"]}),
            ],
            # Maestro NME:
            "NMA": [
                ResidueDefinition.from_smiles(
                    residue_name="NMA",
                    mapped_smiles="[H:3][N:1]([H:4])[C:2]([H:5])([H:6])[H:7]",
                    atom_names={
                        1: "N",
                        2: "C",
                        3: "HN1",
                        4: "HN2",
                        5: "H1",
                        6: "H2",
                        7: "H3",
                    },
                    leaving_atoms=[3],
                    linking_bond=PEPTIDE_BOND,
                    description="METHYLAMINE (MAESTRO)",
                ).with_synonyms(
                    {
                        "HN2": ["H"],
                        "C": ["CA", "CH3"],
                        "H1": ["1HH3", "HA1", "1HA"],
                        "H2": ["2HH3", "HA2", "2HA"],
                        "H3": ["3HH3", "HA3", "3HA"],
                    },
                ),
            ],
        },
        fail_on_error=False,  # Be tolerant to errors so users don't get importtime errors
    )


STD_CCD_CACHE: CcdCache = _construct_std_ccd_cache()
