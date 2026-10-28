import importlib.resources
from functools import lru_cache
from pathlib import Path

from openff.pablo._matching import ResidueMatch
from openff.pablo._utils import unwrap
from openff.pablo.residue import AtomDefinition, ResidueDefinition


@lru_cache
def get_test_data_path(path: Path | str) -> Path:
    """Get the filename of a resource"""
    # TODO: Do this non-deprecated-ly
    assert __package__ is not None
    with importlib.resources.as_file(importlib.resources.files(__package__)) as root:
        pass
    assert root.is_dir()
    return root / "data" / path


def only_match(
    residue_definition: ResidueDefinition,
    index_to_atomdef: dict[int, AtomDefinition],
) -> ResidueMatch:
    """
    Match ``index_to_atomdef`` to ``residue_definition`` and return the match.

    Matching gives one match for each combination of bonds to other residues
    that the missing atoms allow, plus mismatches for the combinations they
    don't. This asserts that there is exactly one match, and returns it typed
    as a ``ResidueMatch`` so tests can use it without further checks.
    """
    return unwrap(
        match
        for match in ResidueMatch.create_split(residue_definition, index_to_atomdef)
        if isinstance(match, ResidueMatch)
    )
