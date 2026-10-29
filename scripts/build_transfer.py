"""Package an explicit source allowlist and SHA256 manifest; never include datasets."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("outputs/audit/ablation_review_manifest.json")


def payload_files(root):
    names = {"AGENTS.md", "README.md", "Architecture.md", "pyproject.toml",
             "environment.yml", "requirements-lock-windows-cpu.txt",
             "outputs/audit/ablation_outputs_review.md"}
    for pattern in ("src/**/*.py", "scripts/*.py", "tests/*.py", "configs/*.yaml", "docs/*.md"):
        names.update(path.relative_to(root).as_posix() for path in root.glob(pattern))
    paths = [root / name for name in sorted(names)]
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("Missing file or symlink in source payload")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError("Use a new archive path inside the project")
    paths = payload_files(ROOT)
    manifest = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    (ROOT / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths + [ROOT / MANIFEST]:
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == set(manifest) | {MANIFEST.as_posix()}
        assert archive.testzip() is None
        for name, digest in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(output), "sha256": digest, "payload_files": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
