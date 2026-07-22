from __future__ import annotations

from posemodel.windows import build_window_index, contiguous_split


def test_contiguous_split_has_guard_gaps_and_no_crossing_windows() -> None:
    intervals = contiguous_split(1000, validation_fraction=0.2, test_fraction=0.2, gap=100)
    assert intervals[1].start - intervals[0].stop == 100
    assert intervals[2].start - intervals[1].stop == 100
    windows = build_window_index(intervals, window_size=100, stride=25)
    for window in windows:
        interval = next(item for item in intervals if item.split == window.split)
        assert interval.start <= window.start < window.stop <= interval.stop
