"""Re-identification evaluation utilities."""

from .reid import ReIDMetrics, evaluate_reid, l2_normalize, summarize_conditions

__all__ = ["ReIDMetrics", "evaluate_reid", "l2_normalize", "summarize_conditions"]
