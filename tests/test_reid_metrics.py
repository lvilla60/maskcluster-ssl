import numpy as np
import pytest

from maskcluster_ssl.evaluation.reid import evaluate_reid, l2_normalize, summarize_conditions


def test_known_rankings_exclusions_and_query_without_positive():
    gallery = np.asarray([[1, 0], [0, 1], [0.8, 0.2], [-1, 0], [0.5, 0.5], [-0.2, 0.1]], dtype=np.float32)
    query = np.asarray([[0.3, 1.0], [-1, -1], [1, 0]], dtype=np.float32)
    metrics = evaluate_reid(
        query,
        gallery,
        query_person_ids=np.asarray([1, 3, 9]),
        gallery_person_ids=np.asarray([1, 2, 1, 3, -1, 4]),
        query_camera_ids=np.asarray([1, 1, 1]),
        gallery_camera_ids=np.asarray([2, 1, 1, 2, 3, 1]),
        query_block_size=1,
    )
    assert metrics.valid_queries == 2
    assert metrics.excluded_queries == 1
    assert metrics.mAP == pytest.approx(0.75)
    assert metrics.rank1 == pytest.approx(0.5)
    assert metrics.rank5 == pytest.approx(1.0)
    assert metrics.rank10 == pytest.approx(1.0)


def test_block_size_does_not_change_metrics_and_delta_is_percentage_points():
    rng = np.random.default_rng(7)
    query = rng.normal(size=(4, 6))
    gallery = rng.normal(size=(8, 6))
    q_pid = np.asarray([1, 2, 3, 4])
    g_pid = np.asarray([1, 2, 3, 4, 1, 2, 3, 4])
    q_cam = np.ones(4, dtype=int)
    g_cam = np.asarray([2, 2, 2, 2, 1, 1, 1, 1])
    one = evaluate_reid(query, gallery, q_pid, g_pid, q_cam, g_cam, query_block_size=1)
    all_at_once = evaluate_reid(query, gallery, q_pid, g_pid, q_cam, g_cam, query_block_size=8)
    assert one == all_at_once
    low = type(one)(mAP=one.mAP - 0.125, rank1=one.rank1, rank5=one.rank5,
                    rank10=one.rank10, valid_queries=one.valid_queries,
                    excluded_queries=one.excluded_queries)
    summary = summarize_conditions(one, low)
    assert summary["mAP_drop_percentage_points"] == pytest.approx(12.5)
    assert summary["mean_mAP"] == pytest.approx(one.mAP - 0.0625)


def test_normalization_rejects_zero_vectors():
    with pytest.raises(ValueError, match="zero"):
        l2_normalize(np.zeros((1, 3), dtype=np.float32))
