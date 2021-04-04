import numpy as np
import soundfile as sf
import keras
from keras import models
from ML.FeatureExtraction import custom_layers
import os


def get_mono(file_path):
    """
    Returns the data in a given file after converting it to mono channel.
    """
    with sf.SoundFile(file_path, "r") as f:
        a = f.read(always_2d=True)
        sr = f.samplerate
    a = np.mean(a, -1)
    return a, sr


def get_data(file_path, interval):
    start, end = interval

    a, sr = get_mono(file_path)

    start_t = int(sr * start)
    # end_t = int(sr * start)
    # length = int(start_t - end_t)
    length = 144000
    end_t = start_t + length

    if end_t > len(a):
        start_t = len(a) - length
        if start_t < 0:
            # Too short, padding
            pad_r = length - len(a)
            pad_l = pad_r // 2
            pad_r -= pad_l
            return np.pad(a, pad_width=(pad_l, pad_r))

    return a[start_t:end_t]


def run_model(file_path, interval, model):
    input_data = get_data(file_path, interval)
    input_data = np.reshape(input_data, (1, 144000))
    return model.predict(input_data, batch_size=1)


def get_activations(birdnet_input):
    h5file = os.path.join(
        os.path.dirname(__file__),
        "BirdNET_1000_RAW_model.h5",
    )
    model = models.load_model(
        h5file,
        custom_objects={"SimpleSpecLayer": custom_layers.SimpleSpecLayer},
        compile=False,
    )
    model.compile()
    model2 = keras.models.Model(
        inputs=model.input, outputs=model.get_layer(index=-3).output
    )
    model2.compile()
    result = []
    for (file_name, intervals) in birdnet_input:
        for t in intervals:
            result.append((file_name, t, run_model(file_name, t, model2)))
    return result
