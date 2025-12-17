import math
from collections.abc import Iterable

import gemmi

from openff.pablo._utils import flatten
from openff.pablo.exceptions import PabloError


def cif_float(s: str) -> float:
    """Convert a CIF value to a float; raise error if missing or invalid"""
    n = gemmi.cif.as_number(s)
    if math.isnan(n):
        raise PabloError(f"cannot convert {s!r} to float")
    return n


def cif_str(s: str) -> str:
    """Convert a CIF value to a str; return empty string if missing or invalid"""
    return gemmi.cif.as_string(s)


def cif_int(s: str) -> int:
    """Convert a CIF value to an int; raise error if missing or invalid"""
    return gemmi.cif.as_int(s)


def cif_opt_float(s: str) -> float | None:
    """Convert a CIF value to a float; return None if missing or invalid"""
    try:
        return cif_float(s)
    except ValueError:
        return None


def cif_opt_str(s: str) -> str | None:
    """Convert a CIF value to a str; return None if missing or invalid"""
    try:
        s = cif_str(s)
    except ValueError:
        return None
    if s == "":
        return None


def cif_opt_int(s: str) -> int | None:
    """Convert a CIF value to an int; return None if missing or invalid"""
    try:
        return gemmi.cif.as_int(s)
    except ValueError:
        return None


def cif_floats(strs: Iterable[str]) -> list[float]:
    """Convert CIF values to floats; raise error if any missing or invalid"""
    return [cif_float(s) for s in strs]


def cif_strs(strs: Iterable[str]) -> list[str]:
    """Convert CIF values to strs; return empty string if missing or invalid"""
    return [cif_str(s) for s in strs]


def cif_ints(strs: Iterable[str]) -> list[int]:
    """Convert CIF values to ints; raise error if any missing or invalid"""
    return [cif_int(s) for s in strs]


def cif_opt_floats(strs: Iterable[str]) -> list[float | None]:
    """Convert CIF values to floats; return None if any missing or invalid"""
    return [cif_opt_float(s) for s in strs]


def cif_opt_strs(strs: Iterable[str]) -> list[str | None]:
    """Convert CIF values to strs; return None if any missing or invalid"""
    return [cif_opt_str(s) for s in strs]


def cif_opt_ints(strs: Iterable[str]) -> list[int | None]:
    """Convert CIF values to ints; return None if any missing or invalid"""
    return [cif_opt_int(s) for s in strs]


def parse_cif(string: str) -> list[dict[str, list[str]]]:
    """Parse a CIF file into a boring Python data structure.

    The outer list has one element per block. Each block is represented as a
    dict mapping from tags to lists of values. Simple pairs are returned as
    lists with length 1. Loops are returned as one list for each tag. All
    tags include the entire path to the leaf tag. Each tag has a subtag
    ``.__pablo__line_no`` which stores the line number(s) for that tag's values.
    Frames are not supported.
    """
    document = gemmi.cif.read_string(string)
    blocks: list[dict[str, list[str]]] = []
    for block in document:
        blocks.append({})
        for item in block:
            if item.loop is not None:  # pyright: ignore[reportUnnecessaryComparison]
                loop = item.loop
                # item.line_number is incorrect - appears to be 2n+1 where n is
                # the true line number
                # item.line_number is supposed to be the line number of the start
                # of the loop - between that and the first row, there are 2 lines
                # of syntax and loop.width() lines of column tags
                first_row_line_no = (item.line_number // 2) + loop.width() + 2
                tags = list(
                    flatten((tag, tag + ".__pablo__line_no") for tag in loop.tags),
                )
                if any(tag in blocks[-1] for tag in tags):
                    raise PabloError("tag cannot be set twice")
                blocks[-1].update(
                    {
                        tag: [loop[j, i] for j in range(loop.length())]
                        for i, tag in enumerate(loop.tags)
                    },
                )
                blocks[-1].update(
                    {
                        tag + ".__pablo__line_no": [
                            str(first_row_line_no + j) for j in range(loop.length())
                        ]
                        for tag in loop.tags
                    },
                )
            elif item.pair is not None:
                tag, value = item.pair  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]
                if tag in blocks[-1] or tag + ".__pablo__line_no" in blocks[-1]:
                    raise PabloError("tag cannot be set twice")
                blocks[-1][tag] = [value]
                blocks[-1][tag + ".__pablo__line_no"] = [str(item.line_number // 2)]
            elif item.frame is not None:
                raise PabloError("frames not supported")
            else:
                raise PabloError("unknown item type")

    return blocks
