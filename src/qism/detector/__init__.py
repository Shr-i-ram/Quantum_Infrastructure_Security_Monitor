"""Detector array components: SPAD model, dark counts, timing."""

from qism.detector.dark_counts import dark_click_probability
from qism.detector.detector import Detector
from qism.detector.timing import sample_dark_arrival_offset, sample_signal_arrival_offset

__all__ = [
    "Detector",
    "dark_click_probability",
    "sample_dark_arrival_offset",
    "sample_signal_arrival_offset",
]
