from math import floor
import numpy as np


def convert_ts_format(time):
    """
    Converts hh:mm:ss.ss strings into the equivalent number of seconds (expressed as a float).

    Parameters
    ----------
    time : str

    Returns
    -------
    float

    """
    hour, minute, second = [float(x) for x in time.split(":")]
    return 3600 * hour + 60 * minute + second


def parse_detections(detections):
    """
    Parses the timestrings into floats across all tuples in detections.

    Parameters
    ----------
    detections : list of tuples of (str, list of str)

    Returns
    -------
    detections_parsed : list of tuples of (str, list of float)
    """

    detections_parsed = []
    for (filename, timestamps) in detections:
        detections_parsed.append(
            [filename, [convert_ts_format(timestamp) for timestamp in timestamps]]
        )
    return detections_parsed


def build_detection_map(detections):
    """
    Builds a map from filenames to a set of integers, where each integer represents
    a 1 second timeframe containing where the bird call was detected according to BirdVox.
    Simply uses floor function.

    Parameters
    ----------
    detections : list of tuples of (str, list of float)
        Each tuple is a pair (filename, detection_list) where each element in detection_list
        represents a moment where a bird call was detected.

    Returns
    -------
    detection_map : dict from str to set of int
    """
    detection_map = {}
    for (filename, timestamps) in detections:
        if filename not in detection_map:
            detection_map[filename] = set()
        for timestamp in timestamps:
            detection_map[filename].add(floor(timestamp))
    return detection_map


def get_intervals(timestamps, length=3):
    """
    Returns the minimum interval set that captures all time detections.

    Parameters
    ----------
    timestamps : list of float

    length : int

    Returns
    -------
    intervals : list of pairs of (float, float)
    """

    intervals = []
    itr = 0
    ref_itr = 0

    while itr < len(timestamps):

        startTimeStamp = timestamps[ref_itr]

        while itr < len(timestamps) and timestamps[itr] <= startTimeStamp + length:
            itr += 1

        endTimeStamp = timestamps[itr - 1]
        midTimeStamp = (startTimeStamp + endTimeStamp) / 2

        # Check for zeros
        if midTimeStamp - length / 2 < 0:
            intervals.append((0, length))
        else:
            intervals.append((midTimeStamp - length / 2, midTimeStamp + length / 2))

        ref_itr = itr

    return intervals


def get_timestamps(detections, length=3):
    """
    Creates interval timestamps based on precise moments that have been tagged
    to contain a bird call by BirdVox.

    Parameters
    ----------
    detections : list of tuples of (str, list of str)
        Each tuple is a pair (filename, list of times in format hh:mm:ss.ss
        where a bird call was detected in filename). Note that for each tuple, the list of pairs
        must be sorted.

    length : int
        The length of each audio slice we wish to create.

    Returns
    -------
    detection_map : dict from str to set of int
        For each filename, a set of integers contains elements n (with unit seconds)
        such that [n, n + 1) has a bird call detected in it.

    birdnet_input : list of tuples of (str, list of tuples of (float, float))
        Each tuple is a pair (filename, list containing pairs (start, end) timestamps to be used in
        BirdNET feature extraction.)
    """

    birdnet_input = []
    detections_parsed = parse_detections(detections)
    detection_map = build_detection_map(detections_parsed)

    for (filename, timestamps) in detections_parsed:
        intervals = get_intervals(timestamps)
        birdnet_input.append((filename, intervals))

    return detection_map, birdnet_input
