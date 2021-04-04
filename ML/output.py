import pandas as pd
import os
import numpy as np
from math import floor
from math import ceil
import soundfile as sf
import ML.Utils.InvalidParameterError as InvalidParameterError


def convert_to_letters(num):
    """Converts number to string of capital letters in Excel column name
    fashion, i.e. 1 -> A, 26 -> Z, 27 -> AA ...."""

    string_equiv = ""
    while num > 0:
        currNum = (num - 1) % 26
        num = (num - 1) // 26
        string_equiv = chr(currNum + ord("A")) + string_equiv
    return string_equiv


def get_file_name(file_path, timestamp, outDir):
    file_name_without_ext = os.path.split(file_path)[-1] + "_Timestamp_{}".format(timestamp)
    file_name_without_ext = os.path.join(outDir, file_name_without_ext)
    file_extension = os.path.splitext(file_path)[-1]
    outpath = file_name_without_ext + file_extension

    i = 0
    while os.path.exists(outpath):
        i += 1
        outpath = file_name_without_ext + "({})".format(i) + file_extension

    with sf.SoundFile(file_path, "r") as f:
        sr = f.samplerate
        channels = f.channels
        start = int(max((timestamp - 2.5) * sr, 0))
        end = int(min((timestamp + 2.5) * sr, len(f)))
        f.seek(start)
        data = f.read(frames=(end - start), always_2d=True)

    with sf.SoundFile(outpath, "w", samplerate=sr, channels=channels) as f:
        f.write(data)

    return outpath


def create_flutter_output(outFrame, outDir):
    """
    Auxiliary Function to create output for Flutter Frontend.
    """
    map_from_type_to_files_with_secs = {}
    for i in outFrame.index:
        if outFrame["CallType"][i] not in map_from_type_to_files_with_secs.keys():
            map_from_type_to_files_with_secs[outFrame["CallType"][i]] = []
        map_from_type_to_files_with_secs[outFrame["CallType"][i]].append(
            (outFrame["FullPath"][i], outFrame["SecondDetected"][i])
        )
    final_output = {
        "CallType": [],
        "Number": [],
        "File1": [],
        "File2": [],
        "File3": [],
        "File4": [],
        "File5": [],
    }
    FileKeys = ["File1", "File2", "File3", "File4", "File5"]
    for callType in map_from_type_to_files_with_secs.keys():
        final_output["CallType"].append(callType)
        final_output["Number"].append(str(len(map_from_type_to_files_with_secs[callType])))
        if len(map_from_type_to_files_with_secs[callType]) < 5:
            for i in range(len(map_from_type_to_files_with_secs[callType])):
                file_name = get_file_name(
                    *(map_from_type_to_files_with_secs[callType][i]), outDir
                )
                final_output[FileKeys[i]].append(file_name)
            for i in range(len(map_from_type_to_files_with_secs[callType]), 5):
                final_output[FileKeys[i]].append(" ")
        else:
            idx = np.random.choice(
                len(map_from_type_to_files_with_secs[callType]), 5, replace=False
            )
            for i in range(5):
                file_name = get_file_name(
                    *(map_from_type_to_files_with_secs[callType][idx[i]]), outDir
                )
                final_output[FileKeys[i]].append(file_name)
    
    final_output = pd.DataFrame.from_dict(final_output)
    final_output.sort_values(by=["CallType"], ascending=True, inplace=True)
    output_path = os.path.join(outDir, "flutter_aux_output.csv")
    final_output.to_csv(output_path, encoding="UTF-8", index=False)


def create_output(filenames, clusters, timestamps, output_path, detection_map, outDir):
    """
    Saves information about bird clusters and when their audio can be heard into a csv file, with headers Filename, CallType, and SecondDetected.

    Parameters
    ----------
    filenames: list of str
        Contains the filenames where each cluster was extracted from.

    timestamps: list of tuples of int
        Contains timestamps described as a pair (start, end) in seconds where bird calls were detected.

    clusters: list of int
        Contains cluster_ID assigned to each pair of timestamps. It is assumed that timestamps[i] corresponds to clusters[i]

    output_path : str
        The path where we wish to save the output. Must contain the CSV file name.

    detection_map : dict from str to set of int
        For each filename, a set of integers contains elements n (with unit seconds)
        such that [n, n + 1) has a bird call detected in it.

    outDir : str
        Path to the output directory where the Audio Samples will be stored.

    Returns
    -------
    None

    Raises
    ------
    InvalidParameterError
        Raised when one-to-one correspondences between filenames, clusters, and timestamps
        cannot be made, or when output_path is a directory or already exists.
    """

    if not (len(filenames) == len(clusters) and len(clusters) == len(timestamps)):
        raise InvalidParameterError(
            "Number of files, clusters, and timestamps are not equal. Cannot match correctly. Program terminating."
        )

    if os.path.exists(output_path) and os.path.isdir(output_path):
        raise InvalidParameterError(
            "Please indicate a filename rather than a directory. Program terminating."
        )

    calls = {"Filename": [], "CallType": [], "SecondDetected": [], "FullPath": []}

    filenames_relative = [os.path.split(x)[-1] for x in filenames]

    for idx in range(len(timestamps)):
        currFilename = filenames[idx]
        currFilename_relative = filenames_relative[idx]
        currCluster = clusters[idx] + 1
        start_time, end_time = floor(timestamps[idx][0]), ceil(timestamps[idx][1])

        for currTime in range(start_time, end_time):
            if currTime in detection_map[currFilename]:
                calls["Filename"].append(currFilename_relative)
                calls["SecondDetected"].append(currTime)
                calls["CallType"].append(currCluster)
                calls["FullPath"].append(currFilename)

    output = pd.DataFrame.from_dict(calls)
    output.sort_values(
        by=["Filename", "CallType", "SecondDetected"], ascending=True, inplace=True
    )
    output["CallType"] = output["CallType"].apply(lambda num: convert_to_letters(num))
    output["CallType"] = "Type " + output["CallType"]
    output.drop_duplicates(inplace=True)

    create_flutter_output(output, outDir)
    del output["FullPath"]

    output.to_csv(output_path, encoding="UTF-8", index=False)
