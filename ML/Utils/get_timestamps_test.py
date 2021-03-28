from ML.Utils import get_timestamps
import numpy as np
import pytest
from math import floor


def convert_ts_format_test(timestamps_to_test, expected_answers):
    # Check that conversions are as expected when done by hand.
    for i in range(len(timestamps_to_test)):
        assert (
            abs(
                get_timestamps.convert_ts_format(timestamps_to_test[i])
                - expected_answers[i]
            )
            < 1e-3
        )
    print("convert_ts_format has passed all {} tests.".format(len(timestamps_to_test)))


def get_intervals_test(timestamps, length=3):
    resulting_windows = get_timestamps.get_intervals(timestamps, length)

    # Check that all windows are of size length, and no windows span negative starts/ends
    for (start, end) in resulting_windows:
        assert abs((end - start) - length) < 1e-3
        assert start >= 0
        assert end >= 0

    # Check that each timestamp is present in a window
    for timestamp in timestamps:
        assert any(
            [
                timestamp >= start and timestamp <= end
                for (start, end) in resulting_windows
            ]
        )

    print("get_intervals has passed all tests.")


def build_detection_map_test(detections):
    detection_map = get_timestamps.build_detection_map(detections)

    for (filename, timestamps) in detections:
        set_of_floors = set()
        for timestamp in timestamps:
            # Ensure that all timestamps are represented in the detection map
            assert floor(timestamp) in detection_map[filename]
            set_of_floors.add(floor(timestamp))

        # Ensure that there is no superfluous timestamps in detection map
        assert len(set_of_floors) == len(detection_map[filename])

    print("build_detection_map has passed all tests.")


def get_timestamps_test(detections, length=3):
    detection_map, filenames_with_intervals = get_timestamps.get_timestamps(detections)
    print("get_timestamps has compiled successfully.")


convert_ts_format_test(
    timestamps_to_test=[
        "12:32:52.15",
        "06:17:42.12",
        "00:00:00.00",
        "12:00:00.00",
        "00:52:42.32",
    ],
    expected_answers=[45172.15, 22662.12, 0, 43200, 3162.32],
)

get_intervals_test(sorted(list(np.random.uniform(0, 30, 100))))

build_detection_map_test(
    [
        ["A", list(np.random.uniform(0, 30, 100))],
        ["B", list(np.random.uniform(0, 30, 100))],
        ["C", list(np.random.uniform(0, 30, 100))],
        ["D", list(np.random.uniform(0, 30, 100))],
    ]
)

get_timestamps_test(
    [
        [
            "A",
            [
                "00:00:00.00",
                "00:52:42.32",
                "06:17:42.12",
                "12:00:00.00",
                "12:32:52.15",
            ],
        ],
        [
            "B",
            [
                "00:00:00.00",
                "00:52:42.32",
                "06:17:42.12",
                "12:00:00.00",
                "12:32:52.15",
            ],
        ],
        [
            "C",
            [
                "00:00:00.00",
                "00:52:42.32",
                "06:17:42.12",
                "12:00:00.00",
                "12:32:52.15",
            ],
        ],
        [
            "D",
            [
                "00:00:00.00",
                "00:52:42.32",
                "06:17:42.12",
                "12:00:00.00",
                "12:32:52.15",
            ],
        ],
    ]
)
