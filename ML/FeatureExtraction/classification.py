import ML.FeatureExtraction.model.initialisation as i
import numpy as np
from os import path
from pydub import AudioSegment
from ML.FeatureExtraction import birdNET_preprocess


model_path = path.join(path.dirname(__file__), "model/birdNet")
model = i.load_model(model_path)


def get_spectrograms(result_list):

    """

    function for getting the spectograms given input audio segments.

    Parameters
    ----------
    result_list: list of tuple (str, list of tuple (float,float) )
        This is the output of the bird detection stage of the pipeline, this is a
        list of (file,timestamp_list) pairs. With file being the path to some input audio file,
        and the timestamp list being float pairs describing the range in which bird calls were
        detected within the file.

    Returns
    -------
    file_ts_act_list: list of numpy_array
        This is a list of features , which we get from feeding the spectrograms into the
        birdNet model. The returned spectograms correspond in order with the input.
    """

    file_ts_spec_list = []
    for (filename, timestamps) in result_list:
        audio = AudioSegment.from_wav(filename)
        for timestamp, spec in birdNET_preprocess.specsFromTimestamps(
            audio, timestamps
        ):
            file_ts_spec_list.append((filename, timestamp, spec))
    return file_ts_spec_list


def get_activations(result_list):
    """

    function for getting features given input audio segments.

    Parameters
    ----------
    result_list: list of tuple (str, list of tuple (float,float) )
        This is the output of the bird detection stage of the pipeline, this is a
        list of (file,timestamp_list) pairs. With file being the path to some input audio
        file, and the timestamp list being float pairs describing the range in which bird
        calls were detected within the file.

    Returns
    -------
    file_ts_act_list: list of numpy_array
        This is a list of features , which we get from feeding the spectrograms into the
        birdNet model. The returned spectograms correspond in order with the input.

    """
    # contains tuples of (filename, timestamp, spec)
    file_ts_spec_list = get_spectrograms(result_list)

    file_ts_act_list = []
    for (f, t, s) in file_ts_spec_list:
        file_ts_act_list.append(
            (f, t, model.run_model(np.reshape(s, (1, 1, 64, 384))))
        )  # make sure this is right

    return file_ts_act_list
