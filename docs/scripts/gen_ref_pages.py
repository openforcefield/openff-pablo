"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files
import griffe

nav = mkdocs_gen_files.Nav()
# Collect object navigations seperately so that modules come before objects
obj_nav = {}
src = Path(__file__).parent.parent.parent / "openff"

pkg_griffe = griffe.load(src)
mod_symbol = '<code class="doc-symbol doc-symbol-nav doc-symbol-module"></code>'


def is_private_module(path: Path) -> bool:
    if path.name == "__init__.py":
        parts = path.parts[:-1]
    else:
        parts = path.parts

    for part in parts:
        if part.startswith("_"):
            return True
    return False


# Iterate over python modules
for path in sorted(src.rglob("*.py")):
    # Skip private modules
    if is_private_module(path):
        continue

    # Sort out where we're reading from and writing to
    module_path = path.relative_to(src.parent).with_suffix("")
    doc_path = path.relative_to(src)
    parts = tuple(module_path.parts)

    # Choose the appropriate index.md to write to
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
    else:
        doc_path = doc_path / "index.md"
    full_doc_path = Path("reference", doc_path)

    # Get the public members of this module
    mod_griffe = pkg_griffe[parts[1:]]
    public_names = [n for n in mod_griffe.all_members if (mod_griffe[n].is_public)]

    # Add this module to the navigation
    nav_parts = [f"{mod_symbol} {part}" for part in parts]
    nav[tuple(nav_parts)] = doc_path.as_posix()

    # Write the stub for this module
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        mod_ident = ".".join(parts)
        fd.write(f"::: {mod_ident}\n")
        # fd.write("    options:\n")
        # fd.write(f"      summary: true\n")
        # fd.write(f"      members: {public_names}\n")
        # fd.write(f"      members: []\n")

    # Write stubs for the members of this module
    for name in public_names:
        # Make sure we don't collide filenames with the module index
        if "index" in public_names and name.startswith("index"):
            filename = name + "_"
        else:
            filename = name
        obj_doc_path = doc_path.with_name(f"{filename}.md")
        obj_full_doc_path = full_doc_path.with_name(f"{filename}.md")

        obj_griffe = mod_griffe[name]

        # Skip this member if its a module - it's already been documented
        if obj_griffe.kind == griffe.Kind.MODULE:
            continue

        # Add this object to the navigation
        symbol = f'<code class="doc-symbol doc-symbol-nav doc-symbol-{obj_griffe.kind.value}"></code>'
        obj_nav[*nav_parts, f"{symbol} {name}"] = obj_doc_path.as_posix()

        with mkdocs_gen_files.open(obj_full_doc_path, "w") as fd:
            ident = f"{mod_ident}.{name}"
            fd.write(f"::: {ident}\n")

    # Not quite sure what this does
    mkdocs_gen_files.set_edit_path(full_doc_path, ".." / path)

# All the modules have been added, time to add the objects so they come after
for k, v in obj_nav.items():
    nav[k] = v

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
