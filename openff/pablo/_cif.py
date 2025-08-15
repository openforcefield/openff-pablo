from collections.abc import Iterable

import gemmi


def cif_float(s: str) -> float:
    return gemmi.cif.as_number(s)


def cif_str(s: str) -> str:
    return gemmi.cif.as_string(s)


def cif_int(s: str) -> int:
    return gemmi.cif.as_int(s)


def cif_opt_float(s: str) -> float | None:
    try:
        return gemmi.cif.as_number(s)
    except ValueError:
        return None


def cif_opt_str(s: str) -> str | None:
    try:
        s = gemmi.cif.as_string(s)
    except ValueError:
        return None
    if s == "":
        return None


def cif_opt_int(s: str) -> int | None:
    try:
        return gemmi.cif.as_int(s)
    except ValueError:
        return None


def cif_floats(strs: Iterable[str]) -> list[float]:
    return [cif_float(s) for s in strs]


def cif_strs(strs: Iterable[str]) -> list[str]:
    return [cif_str(s) for s in strs]


def cif_ints(strs: Iterable[str]) -> list[int]:
    return [cif_int(s) for s in strs]


def cif_opt_floats(strs: Iterable[str]) -> list[float | None]:
    return [cif_opt_float(s) for s in strs]


def cif_opt_strs(strs: Iterable[str]) -> list[str | None]:
    return [cif_opt_str(s) for s in strs]


def cif_opt_ints(strs: Iterable[str]) -> list[int | None]:
    return [cif_opt_int(s) for s in strs]


def parse_cif(string: str) -> list[dict[str, list[str]]]:
    document = gemmi.cif.read_string(string)
    blocks: list[dict[str, list[str]]] = []
    for block in document:
        blocks.append({})
        for item in block:
            if item.loop is not None:  # type: ignore
                loop = item.loop
                if any(tag in blocks[-1] for tag in loop.tags):
                    raise ValueError("tag cannot be set twice")
                blocks[-1].update(
                    {
                        tag: [loop[j, i] for j in range(loop.length())]
                        for i, tag in enumerate(loop.tags)
                    },
                )
            elif item.pair is not None:
                tag, value = item.pair  # type: ignore
                if tag in blocks[-1]:
                    raise ValueError("tag cannot be set twice")
                blocks[-1][tag] = [value]
            elif item.frame is not None:  # type: ignore
                raise ValueError("frames not supported")
            else:
                raise ValueError("unknown item type")

    return blocks
