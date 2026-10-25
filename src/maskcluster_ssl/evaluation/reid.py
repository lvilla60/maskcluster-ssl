"""Market-1501-style retrieval metrics using cosine similarity."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ReIDMetrics:
    mAP: float
    rank1: float
    rank5: float
    rank10: float
    valid_queries: int
    excluded_queries: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def l2_normalize(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize rows and reject zero or non-finite embeddings."""
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Embeddings must be a finite two-dimensional array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("Cannot normalize a zero embedding")
    return values / np.maximum(norms, eps)


def _average_precision(matches: np.ndarray) -> float:
    relevant = np.flatnonzero(matches)
    if relevant.size == 0:
        raise ValueError("AP is undefined without a valid positive")
    precision = np.arange(1, relevant.size + 1, dtype=np.float64) / (relevant + 1)
    return float(precision.mean())


def rank_query(scores, query_pid, query_cam, gallery_pids, gallery_cams, junk_person_id=-1):
    """Return valid gallery positions in stable cosine order and relevance flags."""
    g_pid = np.asarray(gallery_pids)
    g_cam = np.asarray(gallery_cams)
    valid = (g_pid != junk_person_id) & ~((g_pid == query_pid) & (g_cam == query_cam))
    indices = np.flatnonzero(valid)
    ranked = indices[np.argsort(-np.asarray(scores)[indices], kind="stable")]
    return ranked, g_pid[ranked] == query_pid


def evaluate_reid(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_person_ids: np.ndarray,
    gallery_person_ids: np.ndarray,
    query_camera_ids: np.ndarray,
    gallery_camera_ids: np.ndarray,
    *,
    query_block_size: int = 128,
    junk_person_id: int = -1,
) -> ReIDMetrics:
    """Compute mAP and CMC while materializing only a query block at a time.

    Gallery samples with the junk identity and same-identity/same-camera samples
    are removed before ranking. Queries without another-camera positives are
    excluded from every denominator. Stable sorting resolves ties by gallery
    order, making the convention reproducible.
    """
    if query_block_size < 1:
        raise ValueError("query_block_size must be positive")
    query = l2_normalize(query_embeddings)
    gallery = l2_normalize(gallery_embeddings)
    q_pid = np.asarray(query_person_ids)
    g_pid = np.asarray(gallery_person_ids)
    q_cam = np.asarray(query_camera_ids)
    g_cam = np.asarray(gallery_camera_ids)
    if len(query) != len(q_pid) or len(query) != len(q_cam):
        raise ValueError("Query embeddings and metadata are misaligned")
    if len(gallery) != len(g_pid) or len(gallery) != len(g_cam):
        raise ValueError("Gallery embeddings and metadata are misaligned")

    ap_values: list[float] = []
    first_match_ranks: list[int] = []
    excluded = 0
    for start in range(0, len(query), query_block_size):
        similarities = query[start:start + query_block_size] @ gallery.T
        for offset, scores in enumerate(similarities):
            index = start + offset
            if q_pid[index] == junk_person_id:
                excluded += 1
                continue
            ranked_indices, matches = rank_query(scores, q_pid[index], q_cam[index], g_pid, g_cam, junk_person_id)
            if not np.any(matches):
                excluded += 1
                continue
            ap_values.append(_average_precision(matches))
            first_match_ranks.append(int(np.flatnonzero(matches)[0]) + 1)

    if not ap_values:
        raise ValueError("No query has a valid another-camera gallery positive")
    ranks = np.asarray(first_match_ranks)
    return ReIDMetrics(
        mAP=float(np.mean(ap_values)),
        rank1=float(np.mean(ranks <= 1)),
        rank5=float(np.mean(ranks <= 5)),
        rank10=float(np.mean(ranks <= 10)),
        valid_queries=len(ap_values),
        excluded_queries=excluded,
    )


def summarize_conditions(original: ReIDMetrics, low_resolution: ReIDMetrics) -> dict:
    """Report both conditions and absolute mAP degradation in percentage points."""
    return {
        "original": original.to_dict(),
        "low_resolution": low_resolution.to_dict(),
        "mean_mAP": (original.mAP + low_resolution.mAP) / 2.0,
        "mAP_drop_percentage_points": (original.mAP - low_resolution.mAP) * 100.0,
    }
