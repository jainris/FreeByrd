import birdvoxdetect as bvd
import os
import soundfile as sf
import numpy as np
import noisereduce as nr
import pandas as pd


def get_mono(file_path):
    """
    Returns the data in a given file after converting it to mono channel.
    """
    with sf.SoundFile(file_path, "r") as f:
        a = f.read(always_2d=True)
        sr = f.samplerate
    a = np.mean(a, -1)
    return a, sr


def get_temp_name(outdir):
    i = 0
    b = True
    temp_csv_name = "temp{}.csv".format(i)
    temp_wav_name = "temp{}.wav".format(i)
    while b:
        b = False
        while os.path.exists(os.path.join(outdir, temp_csv_name)):
            i += 1
            temp_csv_name = "temp{}.csv".format(i)
        temp_wav_name = "temp{}.wav".format(i)
        if os.path.exists(os.path.join(outdir, temp_wav_name)):
            b = True
            i += 1
    return os.path.join(outdir, temp_wav_name), os.path.join(outdir, temp_csv_name)


def run_birdvoxdetect_segmentation(
    file_names=None,
    dataset_dir=None,
    noise_reduction=False,
    noise_file=None,
    threshold=50.0,
    outdir=None,
):
    """
    Runs birdvoxdetect for getting timestamps of bird calls.

    Parameters
    ----------
    file_names : list of str, optionals
        The list of file names for input.
        If not given then all files in dataset directory are taken.

    dataset_dir : str, optional
        The dataset directory. If not given then it is assumed that file_names contains
        the complete path to the files.

    noise_reduction : boolean, optional
        If True, then noise reduction is done.
        By default set to False.

    noise_file : string, optional
        The path to the file containing the sample expected noise.
        Required if noise_reduction is True.

    threshold : float, optional
        The threshold for birdvoxdetect. A between from 10 to 90, both included.
        Higher value means greater precision at the cost of recall.
        By default set to 50.0.

    Returns
    -------
    pairs : list of tuple (str, list of str, float)
        A list, where each element is a tuple of file_path, timestamps and sample rate,
        where timestamps is a list of timestamps where a bird call is detected,
        as a str in format hh:mm:ss.ss
    """
    found_birds = []
    # giving a random name so as to prevent such a file from existing, and
    # saving it in the same folder as this file. This ensures the checklist
    # generated is also saved in this folder.
    save_path, out_path = get_temp_name(outdir)
    threshold = min(max(threshold, 10.0), 90.0)
    if file_names is None:
        assert (
            dataset_dir is not None
        ), "Need at least one of file_names and dataset_dir"
        file_names = os.listdir(dataset_dir)
    for file_name in file_names:
        file_path = file_name
        if dataset_dir is not None:
            file_path = os.path.join(dataset_dir, file_path)
        wave, samplerate = get_mono(file_path)
        with sf.SoundFile(save_path, "w", samplerate=samplerate, channels=1) as f:
            f.write(wave)
        df = bvd.process_file(save_path, threshold=threshold)
        timestamps = np.array(df["Time (hh:mm:ss)"])
        if noise_reduction:
            assert noise_file is not None, "Need noise file to apply noise reduction"
            with sf.SoundFile(noise_file, "r") as f:
                noise = f.read()
            wave = nr.reduce_noise(wave, noise)
            with sf.SoundFile(save_path, "w", samplerate=samplerate, channels=1) as f:
                f.write(wave)
            df = bvd.process_file(save_path, threshold=threshold)
            timestamps = np.concatenate((timestamps, np.array(df["Time (hh:mm:ss)"])))
            timestamps = np.sort(timestamps, kind="mergesort")
        if timestamps.shape[0] > 0:
            found_birds.append((file_name, timestamps, samplerate))
    if os.path.isfile(save_path):
        os.remove(save_path)
    if os.path.isfile(out_path):
        os.remove(out_path)
    return found_birds
