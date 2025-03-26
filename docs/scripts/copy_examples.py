from pathlib import Path
import mkdocs_gen_files

examples_src_dir = Path(__file__).parent.parent.parent / "examples"
examples_doc_dir = Path("examples")

for path in examples_src_dir.glob("**/*"):
    if not path.is_file():
        continue
    target = examples_doc_dir / path.relative_to(examples_src_dir)
    with mkdocs_gen_files.open(target, "wb") as file:
        file.write(path.read_bytes())
