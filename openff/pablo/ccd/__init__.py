"""
Tools for reading and patching the PDB Chemical Component Dictionary (CCD).

The ``CCD_RESIDUE_DEFINITION_CACHE`` loads, caches, and provides residue
definitions from the CCD and is the default residue database used by
``topology_from_pdb``.
"""

from pathlib import Path

from openff.pablo.residue import ResidueDefinition

from . import patches
from ._ccdcache import CcdCache
from .patches import (
    add_dephosphorylated_5p_terminus,
    add_disulfide_crosslink,
    add_nh2_leaving_atom,
    add_protonation_variants,
    add_synonyms,
    delete_doubly_deprotonated_arginine,
    disambiguate_alt_ids,
    fix_caps,
    patch_his_sidechain_zwitterion,
    set_hop3_leaving,
    strip_linkless_leavers,
)

__all__ = [
    "CCD_RESIDUE_DEFINITION_CACHE",
    "CcdCache",
    "patches",
]

# TODO: Replace these patches with CONECT records?
CCD_RESIDUE_DEFINITION_CACHE: CcdCache = CcdCache(
    # TODO: Use a proper resource setup for this
    library_paths=[Path(__file__).parent / "data/ccd_cache"],
    patches=[
        {
            "ACE": fix_caps,
            "NME": fix_caps,
            "NH2": fix_caps,
            "CYS": add_disulfide_crosslink,
        },
        {"*": add_protonation_variants},
        {
            "U": add_dephosphorylated_5p_terminus,
            "G": add_dephosphorylated_5p_terminus,
            "C": add_dephosphorylated_5p_terminus,
            "A": add_dephosphorylated_5p_terminus,
            "DT": add_dephosphorylated_5p_terminus,
            "DG": add_dephosphorylated_5p_terminus,
            "DC": add_dephosphorylated_5p_terminus,
            "DA": add_dephosphorylated_5p_terminus,
            "NH2": add_nh2_leaving_atom,
        },
        {
            "U": set_hop3_leaving,
            "G": set_hop3_leaving,
            "C": set_hop3_leaving,
            "A": set_hop3_leaving,
            "DT": set_hop3_leaving,
            "DG": set_hop3_leaving,
            "DC": set_hop3_leaving,
            "DA": set_hop3_leaving,
        },
        {"*": disambiguate_alt_ids},
        {"*": add_synonyms},
        {"*": strip_linkless_leavers},
        {
            "HIS": patch_his_sidechain_zwitterion,
            "ARG": delete_doubly_deprotonated_arginine,
        },
    ],
    extra_definitions={
        "I": [ResidueDefinition.from_smiles("[I-:1]", {1: "I"}, "I")],
        "WAT": [
            ResidueDefinition.from_smiles(
                "[H:2][O:1][H:3]",
                {1: "O", 2: "H1", 3: "H2"},
                "WAT",
            ).with_synonyms({"H1": ["1H"], "H2": ["2H"]}),
        ],
    },
)
"""The CCD, with commonly-required patches"""
