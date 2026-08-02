"""Small, CUDA-independent helpers for optional GPU duty-cycle throttling."""

from __future__ import annotations

import argparse


def gpu_util_percent(value):
    """Parse an average GPU duty-cycle target expressed as a percentage."""
    try:
        percent = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("GPU 사용률은 숫자여야 합니다.") from exc
    if not 0.0 < percent <= 100.0:
        raise argparse.ArgumentTypeError(
            "GPU 사용률은 0보다 크고 100 이하여야 합니다."
        )
    return percent


def throttle_delay(active_seconds, target_percent):
    """Return the idle time needed to reach a target average duty cycle."""
    if target_percent >= 100.0:
        return 0.0
    return max(0.0, active_seconds*(100.0/target_percent-1.0))
