"""Sequential, read-only HDF5 data audit. No model inputs are defined here."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw


EXPECTED = {
    "training": {"images": 11043, "camera_ids": 11043, "image_ids": 11043},
    "validation": {
        "query/images": 100, "query/images_low_resolution": 100,
        "query/person_ids": 100, "query/camera_ids": 100, "query/image_ids": 100,
        "gallery/images": 1793, "gallery/person_ids": 1793,
        "gallery/camera_ids": 1793, "gallery/image_ids": 1793,
    },
}
IMAGE_SETS = (("training", "images"), ("validation", "query/images"),
              ("validation", "query/images_low_resolution"),
              ("validation", "gallery/images"))
ISSUE_FIELDS = ("severity", "check", "split", "dataset", "index", "detail")
QUALITY_FIELDS = ("split", "dataset", "index", "shape", "dtype", "channels", "height",
                  "width", "minimum", "maximum", "nan_count", "inf_count", "constant", "black", "status")
DUP_FIELDS = ("kind", "scope", "first_split", "first_dataset", "first_index",
              "second_split", "second_dataset", "second_index", "value")
QUERY_FIELDS = ("index", "person_id", "camera_id", "positive_total", "positive_other_camera",
                "same_identity_same_camera", "valid_for_evaluation")


def within_root(path: str | Path, root: Path) -> Path:
    """Resolve symlinks/junctions and reject paths outside the sole project root."""
    root = root.resolve(strict=True)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path escapes project root: {path}")
    return resolved


def _plain(value):
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", "backslashreplace")
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value[:20]] + (["…"] if value.size > 20 else [])
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _attrs(obj):
    return {key: str(_plain(obj.attrs[key]))[:300] for key in obj.attrs}


def _inventory(file: h5py.File, relative: str, size: int):
    result = {"file": relative, "size_bytes": size, "attributes": _attrs(file), "objects": []}

    def visit(name, obj):
        row = {"path": name, "kind": "dataset" if isinstance(obj, h5py.Dataset) else "group",
               "attributes": _attrs(obj)}
        if isinstance(obj, h5py.Dataset):
            row.update(shape=list(obj.shape), dtype=str(obj.dtype), chunks=list(obj.chunks) if obj.chunks else None,
                       compression=obj.compression)
        result["objects"].append(row)

    file.visititems(visit)
    return result


def _decode(raw):
    if isinstance(raw, (bytes, bytearray, np.bytes_)):
        with Image.open(io.BytesIO(bytes(raw))) as im:
            im.load()
            return np.asarray(im).copy(), f"encoded/{im.format}/{im.mode}"
    if isinstance(raw, np.ndarray) and raw.ndim == 1 and raw.dtype == np.uint8:
        with Image.open(io.BytesIO(raw.tobytes())) as im:
            im.load()
            return np.asarray(im).copy(), f"encoded/{im.format}/{im.mode}"
    if isinstance(raw, np.ndarray):
        return raw, "pixel_array"
    raise ValueError(f"Unsupported image sample type: {type(raw).__name__}")


def _geometry(array, layout):
    if array.ndim == 2:
        return 1, int(array.shape[0]), int(array.shape[1])
    if array.ndim != 3:
        raise ValueError(f"Unsupported image rank {array.ndim}")
    if layout == "NCHW":
        return int(array.shape[0]), int(array.shape[1]), int(array.shape[2])
    if layout == "NHWC":
        return int(array.shape[2]), int(array.shape[0]), int(array.shape[1])
    # Encoded images expose HWC from Pillow; unknown pixel arrays are flagged.
    if layout == "encoded":
        return int(array.shape[2]), int(array.shape[0]), int(array.shape[1])
    raise ValueError("No supported channel layout attribute")


def _digest(array):
    contiguous = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(json.dumps({"shape": list(contiguous.shape), "dtype": contiguous.dtype.str}, sort_keys=True).encode())
    h.update(contiguous.tobytes())
    return h.hexdigest()


def _csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _figure(path, samples, title):
    if not samples:
        return False
    width, height = 190, 225
    canvas = Image.new("RGB", (width * len(samples), height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title[:80], fill="black")
    for j, (array, label, layout) in enumerate(samples):
        arr = array
        if layout == "NCHW" and arr.ndim == 3:
            arr = np.moveaxis(arr, 0, -1)
        if arr.dtype != np.uint8:
            finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
            lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0., 0.)
            arr = np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo)
            arr = np.uint8(np.clip((arr - lo) * (255 / (hi - lo)) if hi > lo else arr * 0, 0, 255))
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        im = Image.fromarray(arr).convert("RGB")
        target_height = 160
        target_width = max(1, round(im.width * target_height / im.height))
        im = im.resize((target_width, target_height), Image.Resampling.NEAREST)
        canvas.paste(im, (j * width + 8, 27))
        draw.multiline_text((j * width + 8, 190), label, fill="black")
    canvas.save(path)
    return True


def audit(data_dir: str | Path, output_dir: str | Path, root: Path, batch_size: int = 64,
          seed: int = 42, progress=print):
    root = root.resolve(strict=True)
    data_dir = within_root(data_dir, root)
    output_dir = within_root(output_dir, root)
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not data_dir.is_dir():
        raise ValueError("data-dir is not a directory")
    paths = {name: within_root(data_dir / f"{name}.h5", root) for name in EXPECTED}
    for p in paths.values():
        if not p.is_file():
            raise ValueError(f"Missing input: {p.name}")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    run = within_root(output_dir / run_id, root)
    run.mkdir(parents=True, exist_ok=False)
    (run / "figures").mkdir()
    issues, quality, duplicates, queries, inventory = [], [], [], [], []
    image_ids = defaultdict(list)
    hashes = defaultdict(list)
    formats = {}
    dimension_counts = defaultdict(Counter)
    cameras = defaultdict(Counter)
    identities = defaultdict(Counter)
    figure_pool = defaultdict(list)
    anomaly_pool = []
    rng = np.random.default_rng(seed)

    def issue(severity, check, split, dataset, index, detail):
        issues.append(dict(severity=severity, check=check, split=split, dataset=dataset,
                           index=index, detail=str(detail)))

    with h5py.File(paths["training"], "r") as train, h5py.File(paths["validation"], "r") as val:
        files = {"training": train, "validation": val}
        for split, file in files.items():
            inventory.append(_inventory(file, str(paths[split].relative_to(root)).replace("\\", "/"), paths[split].stat().st_size))
            for key, expected in EXPECTED[split].items():
                if key not in file or not isinstance(file[key], h5py.Dataset):
                    issue("critical", "schema", split, key, "", "required dataset missing")
                elif file[key].ndim < 1 or file[key].shape[0] != expected:
                    issue("critical", "count", split, key, "", f"expected {expected}, found {file[key].shape}")
            groups = ("",) if split == "training" else ("query/", "gallery/")
            for prefix in groups:
                ikey = prefix + "images"
                if ikey not in file:
                    continue
                n = len(file[ikey])
                for field in ("camera_ids", "image_ids") + (("person_ids",) if split == "validation" else ()):
                    key = prefix + field
                    if key in file and len(file[key]) != n:
                        issue("critical", "metadata_length", split, key, "", f"{len(file[key])} versus images {n}")
                    if key not in file:
                        continue
                    for start in range(0, len(file[key]), batch_size):
                        for offset, value in enumerate(file[key][start:start + batch_size]):
                            idx = start + offset
                            v = _plain(value)
                            if v is None or (isinstance(v, str) and not v.strip()):
                                issue("warning", "empty_id", split, key, idx, "empty or missing ID")
                            if isinstance(v, (int, float)) and not math.isfinite(v):
                                issue("warning", "nonfinite_id", split, key, idx, v)
                            if isinstance(v, (int, float)) and v < 0:
                                issue("warning", "negative_id", split, key, idx, v)
                            if field == "image_ids":
                                image_ids[str(v)].append((split, ikey, idx))
                            elif field == "camera_ids":
                                cameras[f"{split}/{prefix or 'training'}"][str(v)] += 1
                            else:
                                identities[f"{split}/{prefix}"][str(v)] += 1

        for split, key in IMAGE_SETS:
            file = files[split]
            if key not in file:
                continue
            ds = file[key]
            parent = ds.parent
            raw_layout = _plain(ds.attrs.get("layout", file.attrs.get("layout", "")))
            layout = str(raw_layout)
            progress(f"{split}/{key}: {len(ds)} images")
            for start in range(0, len(ds), batch_size):
                end = min(start + batch_size, len(ds))
                try:
                    block = ds[start:end]
                except (OSError, ValueError) as exc:
                    issue("critical", "block_read", split, key, start, f"{type(exc).__name__}: {exc}")
                    # An HDF5 chunk can fail independently; attempt each item.
                    block = None
                for idx in range(start, end):
                    try:
                        raw = block[idx - start] if block is not None else ds[idx]
                        arr, fmt = _decode(raw)
                        actual_layout = "encoded" if fmt.startswith("encoded/") else layout
                        c, h, w = _geometry(arr, actual_layout)
                        if not h or not w or not c or arr.size == 0:
                            raise ValueError("empty image")
                        if not np.issubdtype(arr.dtype, np.number):
                            raise ValueError(f"non-numeric pixels {arr.dtype}")
                        finite = np.isfinite(arr)
                        nan = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
                        inf = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
                        good = arr[finite]
                        lo = _plain(good.min()) if good.size else ""
                        hi = _plain(good.max()) if good.size else ""
                        constant = bool(good.size == arr.size and lo == hi)
                        black = bool(constant and lo == 0)
                        status = "ok"
                        if nan or inf:
                            status = "nonfinite"
                            issue("critical", "nonfinite", split, key, idx, f"NaN={nan}, Inf={inf}")
                        if constant:
                            issue("warning", "constant_image", split, key, idx, f"value={lo}; black={black}")
                        if c not in (1, 3, 4):
                            issue("warning", "channels", split, key, idx, c)
                        if np.issubdtype(arr.dtype, np.integer) and (lo < 0 or hi > 255):
                            issue("warning", "pixel_range", split, key, idx, f"{lo}..{hi}")
                        if np.issubdtype(arr.dtype, np.floating) and (lo < 0 or hi > 1):
                            issue("warning", "float_pixel_range", split, key, idx, f"{lo}..{hi}; scale not assumed")
                        quality.append(dict(split=split, dataset=key, index=idx, shape=str(tuple(arr.shape)),
                                            dtype=str(arr.dtype), channels=c, height=h, width=w, minimum=lo,
                                            maximum=hi, nan_count=nan, inf_count=inf,
                                            constant=constant, black=black, status=status))
                        formats[f"{split}/{key}"] = fmt
                        dimension_counts[f"{split}/{key}"][(c, h, w)] += 1
                        digest = _digest(arr)
                        hashes[digest].append((split, key, idx))
                        # Reservoir sampling limits image memory independent of dataset size.
                        label = f"{split}/{key} #{idx}"
                        if split == "training" and "camera_ids" in parent and idx < len(parent["camera_ids"]):
                            cam = _plain(parent["camera_ids"][idx])
                            label = f"#{idx} cam={cam}\n{w}x{h}"
                            pool_key = f"camera_{cam}"
                        else:
                            pool_key = f"{split}/{key}"
                        pool = figure_pool[pool_key]
                        if len(pool) < 2:
                            pool.append((arr.copy(), label, actual_layout))
                        elif rng.integers(idx + 1) < 2:
                            pool[int(rng.integers(2))] = (arr.copy(), label, actual_layout)
                        if (status != "ok" or constant) and len(anomaly_pool) < 6:
                            anomaly_pool.append((arr.copy(), label, actual_layout))
                    except (OSError, ValueError, TypeError) as exc:
                        issue("critical", "decode", split, key, idx, f"{type(exc).__name__}: {exc}")
                        quality.append(dict(split=split, dataset=key, index=idx, shape="", dtype="", channels="",
                                            height="", width="", minimum="", maximum="", nan_count="",
                                            inf_count="", constant="", black="", status="unreadable"))
            progress(f"  read {len(ds)} / {len(ds)}")

        for digest, refs in hashes.items():
            if len(refs) < 2:
                continue
            for i, first in enumerate(refs):
                for second in refs[i + 1:]:
                    a, b = first[1], second[1]
                    if "low_resolution" in a or "low_resolution" in b:
                        scope = "other_image_sets"
                    elif first[0] != second[0]:
                        scope = "training_validation"
                    elif a == b:
                        scope = "within_" + a.replace("/", "_")
                    else:
                        scope = "query_gallery"
                    duplicates.append(dict(kind="exact_image", scope=scope,
                                           first_split=first[0], first_dataset=a, first_index=first[2],
                                           second_split=second[0], second_dataset=b, second_index=second[2], value=digest))
        for value, refs in image_ids.items():
            if len(refs) > 1:
                for i, first in enumerate(refs):
                    for second in refs[i + 1:]:
                        duplicates.append(dict(kind="image_id", scope="metadata_only", first_split=first[0],
                                               first_dataset=first[1], first_index=first[2],
                                               second_split=second[0], second_dataset=second[1],
                                               second_index=second[2], value=value))

        if all(k in val for k in ("query/images", "query/images_low_resolution")):
            orig, low = val["query/images"], val["query/images_low_resolution"]
            if len(orig) != len(low):
                issue("critical", "low_resolution_count", "validation", "query/images_low_resolution", "", f"{len(low)} versus {len(orig)}")
            for i in range(min(len(orig), len(low))):
                a = next((q for q in quality if q["split"] == "validation" and q["dataset"] == "query/images" and q["index"] == i), None)
                b = next((q for q in quality if q["split"] == "validation" and q["dataset"] == "query/images_low_resolution" and q["index"] == i), None)
                if a and b and a["status"] != "unreadable" and b["status"] != "unreadable":
                    if a["height"] <= b["height"] or a["width"] <= b["width"]:
                        issue("warning", "low_resolution_size", "validation", "query/images_low_resolution", i,
                              "low-resolution dimensions not smaller in both axes")

        if all(k in val for k in ("query/person_ids", "query/camera_ids", "gallery/person_ids", "gallery/camera_ids")):
            gallery_by_person = defaultdict(Counter)
            for i in range(min(len(val["gallery/person_ids"]), len(val["gallery/camera_ids"]))):
                pid = _plain(val["gallery/person_ids"][i]); cam = _plain(val["gallery/camera_ids"][i])
                if isinstance(pid, (int, float)) and pid < 0:
                    continue  # proposed junk convention, not proof of source semantics
                gallery_by_person[str(pid)][str(cam)] += 1
            n = min(len(val["query/person_ids"]), len(val["query/camera_ids"]), len(val["query/images"])) if "query/images" in val else 0
            for i in range(n):
                pid = _plain(val["query/person_ids"][i]); cam = _plain(val["query/camera_ids"][i])
                junk = isinstance(pid, (int, float)) and pid < 0
                matches = gallery_by_person[str(pid)] if not junk else Counter()
                total = sum(matches.values()); same = matches[str(cam)]
                other = total - same
                queries.append(dict(index=i, person_id=pid, camera_id=cam, positive_total=total,
                                    positive_other_camera=other, same_identity_same_camera=same,
                                    valid_for_evaluation=bool(not junk and other)))
                if not junk and not other:
                    issue("warning", "no_valid_positive", "validation", "query/images", i,
                          "no gallery positive from another camera")

    _csv(run / "quality.csv", QUALITY_FIELDS, quality)
    _csv(run / "issues.csv", ISSUE_FIELDS, issues)
    _csv(run / "duplicates.csv", DUP_FIELDS, duplicates)
    _csv(run / "validation_queries.csv", QUERY_FIELDS, queries)
    (run / "inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")
    training_samples = [figure_pool[k][0] for k in sorted(figure_pool) if k.startswith("camera_") and figure_pool[k]]
    _figure(run / "figures" / "training_cameras.png", training_samples[:6], "Training: one sample per camera")
    # The same index is used by the source contract; matching identity is not independently proven.
    pairs = []
    with h5py.File(paths["validation"], "r") as val_fig:
        if all(k in val_fig for k in ("query/images", "query/images_low_resolution")):
            for i in range(min(3, len(val_fig["query/images"]), len(val_fig["query/images_low_resolution"]))):
                for key in ("query/images", "query/images_low_resolution"):
                    try:
                        arr, fmt = _decode(val_fig[key][i])
                        layout = "encoded" if fmt.startswith("encoded/") else str(_plain(val_fig[key].attrs.get("layout", val_fig.attrs.get("layout", ""))))
                        pid = _plain(val_fig["query/person_ids"][i]) if "query/person_ids" in val_fig and i < len(val_fig["query/person_ids"]) else "?"
                        cam = _plain(val_fig["query/camera_ids"][i]) if "query/camera_ids" in val_fig and i < len(val_fig["query/camera_ids"]) else "?"
                        dims = f"{arr.shape[-1]}x{arr.shape[-2]}" if arr.ndim == 3 and layout == "NCHW" else str(tuple(arr.shape))
                        pairs.append((arr, f"#{i} pid={pid} cam={cam}\n{'low' if 'low_resolution' in key else 'original'} {dims}", layout))
                    except (OSError, ValueError, TypeError):
                        pass  # The full scan has already recorded unreadable samples.
    _figure(run / "figures" / "query_pairs.png", pairs[:6], "Query: source-ordered original and low")
    _figure(run / "figures" / "anomalies.png", anomaly_pool, "Image anomalies (examples)")
    counts = {f"{s}/{k}": n for (s, k), n in ((x, len([q for q in quality if q["split"] == x[0] and q["dataset"] == x[1]])) for x in IMAGE_SETS)}
    issue_counts = Counter(x["severity"] for x in issues)
    dup_counts = Counter((x["kind"], x["scope"]) for x in duplicates)
    summary = {
        "counts": counts, "dimensions_chw": {k: [{"chw": list(d), "count": n} for d, n in v.items()] for k, v in dimension_counts.items()},
        "formats": formats, "issue_counts": dict(issue_counts),
        "duplicate_counts": {"/".join(k): v for k, v in dup_counts.items()},
        "camera_frequencies": {k: dict(v) for k, v in cameras.items()},
        "identity_distributions": {k: {"unique": len(v), "negative": sum(n for p, n in v.items() if p.startswith("-")),
                                        "counts_per_identity": dict(Counter(v.values()))} for k, v in identities.items()},
        "valid_queries": sum(q["valid_for_evaluation"] for q in queries), "queries_without_valid_positive": sum(not q["valid_for_evaluation"] for q in queries),
        "positive_other_camera_range": [min((q["positive_other_camera"] for q in queries), default=0), max((q["positive_other_camera"] for q in queries), default=0)],
        "quality_global": {k: {"minimum": min((q["minimum"] for q in quality if f"{q['split']}/{q['dataset']}" == k and q["minimum"] != ""), default=None),
                               "maximum": max((q["maximum"] for q in quality if f"{q['split']}/{q['dataset']}" == k and q["maximum"] != ""), default=None),
                               "constant": sum(bool(q["constant"]) for q in quality if f"{q['split']}/{q['dataset']}" == k),
                               "unreadable": sum(q["status"] == "unreadable" for q in quality if f"{q['split']}/{q['dataset']}" == k)} for k in counts},
    }
    source_root = Path(__file__).resolve().parents[3]
    code_files = (Path(__file__).resolve(), source_root / "scripts/audit_hdf5.py")
    code_hash = hashlib.sha256()
    for source in code_files:
        code_hash.update(source.relative_to(source_root).as_posix().encode())
        code_hash.update(source.read_bytes())
    metadata = {
        "run_id": run_id, "arguments": {"data_dir": str(data_dir.relative_to(root)).replace("\\", "/"),
                                         "output_dir": str(output_dir.relative_to(root)).replace("\\", "/"),
                                         "batch_size": batch_size, "seed": seed},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "h5py", "Pillow", "pandas", "matplotlib", "PyYAML", "pytest")}},
        "code_version": {"package": "maskcluster-ssl 0.1.0", "audit_sources_sha256": code_hash.hexdigest(), "git_commit": None},
        "scope": "Full sequential image, metadata, hash and validation-query checks for training and validation only; figures sampled or first three query pairs.",
        "limitations": ["Low-resolution index pairing is stated by HDF5 metadata, not proven by equal lengths or image content.",
                        "Training identity labels absent: train/validation identity separation cannot be checked.",
                        "Negative person IDs are treated as proposed junk for evaluation viability; source semantics need confirmation.",
                        "No testing file opened. No model metrics or inference."],
        "figure_display": "NCHW transposed to HWC; uint8 unchanged. Non-uint8, if present, scaled from finite per-image min/max solely for figures.",
        "hash_definition": "SHA-256 of JSON shape/dtype.str plus contiguous decoded pixel bytes; no resize or normalization.",
        "results": summary,
    }
    (run / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# HDF5 data audit", "", f"Run: `{run_id}`", "", "## Counts and format", "",
             "| Dataset | Read | Format | CHW distribution |", "| --- | ---: | --- | --- |"]
    for key, n in counts.items():
        lines.append(f"| {key} | {n} | {formats.get(key, 'missing')} | {summary['dimensions_chw'].get(key, [])} |")
    lines += ["", "## Checks", "", f"Critical issues: {issue_counts['critical']}; warnings: {issue_counts['warning']}.",
              f"Exact-image and image-ID duplicate pairs: {dict(dup_counts)}.",
              f"Validation queries with positives from another camera: {summary['valid_queries']}/{len(queries)}.",
              "", "All image and metadata checks were full scans; figures show selected examples.",
              "Hashes compare decoded pixels with shape and dtype, without transforms. Image IDs and identity matches are separate concepts.",
              "Training has no person IDs, so train/validation identity separation cannot be checked.",
              "Low-resolution ordering is stated by source metadata; equal lengths and selected visual pairs cannot prove every correspondence.",
              "Negative person IDs may indicate junk. Proposed evaluator excludes negative gallery IDs and queries with no valid other-camera positive; this requires protocol confirmation.",
              "Constant or black images are warnings, not automatic corruption. Exact query/gallery duplicates are not automatically leakage.",
              "Figures use NCHW to HWC display only; non-uint8 would be scaled for display only.",
              "Testing was not opened. No model evaluation was performed."]
    (run / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run, summary, bool(issue_counts["critical"])
