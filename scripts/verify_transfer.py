"""Verify extracted source files against the transfer manifest without reading data."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "outputs/audit/ablation_review_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or invalid payload path: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"SHA256 mismatch: {name}")
    print(f"Verified {len(manifest)} payload files")


if __name__ == "__main__":
    main()
