"""Deterministic retrieval rankings and inspection panels."""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from maskcluster_ssl.data.hdf5 import ReIDH5Dataset
from .reid import l2_normalize, rank_query


def ranking_records(query_embeddings, gallery_embeddings, query_meta, gallery_meta, *, limit=100):
    query = l2_normalize(query_embeddings)
    gallery = l2_normalize(gallery_embeddings)
    if limit < 1:
        raise ValueError("Ranking limit must be positive")
    for row, scores in enumerate(query @ gallery.T):
        pid = query_meta["person_id"][row]
        cam = query_meta["camera_id"][row]
        if pid == -1:
            continue
        indices, relevant = rank_query(scores, pid, cam, gallery_meta["person_id"], gallery_meta["camera_id"])
        if not relevant.any():
            continue
        hits = []
        for rank, index in enumerate(indices[:limit], 1):
            hits.append({"rank": rank, "gallery_index": int(gallery_meta["index"][index]),
                         "image_id": str(gallery_meta["image_id"][index]),
                         "person_id": str(gallery_meta["person_id"][index]),
                         "camera_id": str(gallery_meta["camera_id"][index]),
                         "similarity": float(scores[index]), "relevant": bool(relevant[rank - 1])})
        yield {"query_index": int(query_meta["index"][row]),
               "image_id": str(query_meta["image_id"][row]),
               "person_id": str(pid), "camera_id": str(cam),
               "rank1_correct": bool(relevant[0]),
               "first_positive_rank": int(np.flatnonzero(relevant)[0]) + 1,
               "valid_gallery_count": len(indices), "ranking": hits}


def write_rankings(path, records):
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _image(array):
    values = np.asarray(array)
    if values.dtype != np.uint8:
        values = (np.clip(values, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(values.transpose(1, 2, 0)).convert("RGB")


def export_cases(validation_path, condition, records, output_dir, *, count=5, columns=5):
    """Export first five valid Rank-1 hits and misses in query index order."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selections = {"success": [], "failure": []}
    for record in records:
        key = "success" if record["rank1_correct"] else "failure"
        if len(selections[key]) < count:
            selections[key].append(record)
    query = ReIDH5Dataset(validation_path, "query", low_resolution=condition == "low_resolution")
    gallery = ReIDH5Dataset(validation_path, "gallery")
    try:
        for key, items in selections.items():
            for position, record in enumerate(items, 1):
                tiles = [("query", query._image(record["query_index"]))]
                for hit in record["ranking"][:columns - 1]:
                    tiles.append((f"#{hit['rank']} {hit['similarity']:.3f} {'+' if hit['relevant'] else '-'}",
                                  gallery._image(hit["gallery_index"])))
                width, height = 128, 256
                panel = Image.new("RGB", (width * len(tiles), height + 48), "white")
                draw = ImageDraw.Draw(panel)
                for column, (label, pixels) in enumerate(tiles):
                    resized = _image(pixels).resize((width, height), Image.Resampling.BICUBIC)
                    panel.paste(resized, (column * width, 48))
                    draw.text((column * width + 4, 4), label, fill="black")
                stem = f"{condition}_{key}_{position:02d}"
                panel.save(output_dir / f"{stem}.png")
                (output_dir / f"{stem}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    finally:
        query.close()
        gallery.close()
    return {key: {"requested": count, "available": len(items)} for key, items in selections.items()}


def write_metrics_csv(path, run_id, variant, encoder, conditions, provenance=None):
    provenance = provenance or {}
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "variant", "encoder", "condition", "mAP", "rank1", "rank5", "rank10", "valid_queries", "excluded_queries", "mAP_drop_percentage_points", *provenance])
        writer.writeheader()
        for name in ("original", "low_resolution"):
            writer.writerow({"run_id": run_id, "variant": variant, "encoder": encoder,
                             "condition": name, **conditions[name],
                             "mAP_drop_percentage_points": conditions["mAP_drop_percentage_points"], **provenance})
