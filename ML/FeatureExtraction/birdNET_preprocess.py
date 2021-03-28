"""Adapted from https://github.com/kahst/BirdNET/blob/master/utils/audio.py. 
Modified several functions for clarity, working with pairs of timestamps 
obtained from previous steps of our model, 
and used AudioSegment module instead of librosa."""

from builtins import range
from pydub import AudioSegment
import numpy as np
import scipy
from scipy import signal

RANDOM = np.random.RandomState(0)
CACHE = {}

# CONSTANTS
IM_DIM = 1
FMIN = 150
FMAX = 15000
OVERLAP = 0
MINLEN = 0.5
LENGTH = 3.0
WIN_LEN = 512
SAMPLE_RATE = 48000
SPEC_TYPE = "melspec"
MAGNITUDE_SCALE = "nonlinear"
IM_SIZE = (64, 384)
BANDPASS = True


def getAudioSlice(audio, audio_tot_len, start_time, end_time):
    """
    Extracts a timeslice from an audio file, and returns its corresponding sample array.
    """
    global SAMPLE_RATE

    # Check if slice goes over the total length of the original clip
    if end_time > audio_tot_len:
        start_time, end_time = (
            max(0, audio_tot_len - (end_time - start_time)),
            audio_tot_len,
        )

    slice = (
        audio[start_time * 1000 : end_time * 1000]
        .set_frame_rate(SAMPLE_RATE)
        .set_channels(1)
    )

    sig = np.array(slice.get_array_of_samples()).astype(np.float)
    rate = slice.frame_rate

    return sig, rate


def noise(sig, shape, amount=None):
    """
    Generates random noise based on a signal.
    """
    # Generate random noise intensity if amount is not specified
    if amount == None:
        amount = RANDOM.uniform(0.1, 0.9)

    # Create Gaussian noise
    noise = RANDOM.normal(min(sig) * amount, max(sig) * amount, shape)

    return noise


def buildBandpassFilter(rate, order=4):
    """
    Creates Bandpass filter for frequency filtering.
    """
    global CACHE

    fname = "bandpass_{}_{}_{}".format(str(rate), str(FMIN), str(FMAX))

    # Check if computation has already been done to avoid expensive recomputing
    if not fname in CACHE:
        wn = np.array([FMIN, FMAX]) / (rate / 2.0)
        filter_sos = scipy.signal.butter(order, wn, btype="bandpass", output="sos")

        # Save to cache
        CACHE[fname] = filter_sos

    return CACHE[fname]


def applyBandpassFilter(sig, rate):
    """
    Applies a bandpass filter to a signal.
    """
    global FMIN
    global FMAX

    # Build filter or load from cache
    filter_sos = buildBandpassFilter(rate)

    return scipy.signal.sosfiltfilt(filter_sos, sig)


def get_mel_filterbanks(num_banks, f_vec, dtype=np.float32):
    """
    Works with an existing vector of frequency bins, as returned from signal.spectrogram(),
    instead of recalculating them and flooring down the bin indices.
    """

    global CACHE
    global FMIN
    global FMAX

    # Avoids expensive recomputing
    fname = "mel_{}_{}_{}".format(str(num_banks), str(FMIN), str(FMAX))
    if not fname in CACHE:

        # Break frequency and scaling factor
        A = 4581.0
        f_break = 1750.0

        # Convert Hz to mel
        freq_extents_mel = A * np.log10(
            1 + np.asarray([FMIN, FMAX], dtype=dtype) / f_break
        )

        # Compute points evenly spaced in mels
        melpoints = np.linspace(
            freq_extents_mel[0], freq_extents_mel[1], num_banks + 2, dtype=dtype
        )

        # Convert mels to Hz
        banks_ends = f_break * (10 ** (melpoints / A) - 1)

        filterbank = np.zeros([len(f_vec), num_banks], dtype=dtype)

        for bank_idx in range(1, num_banks + 1):
            # Points in the first half of the triangle
            mask = np.logical_and(
                f_vec >= banks_ends[bank_idx - 1], f_vec <= banks_ends[bank_idx]
            )
            filterbank[mask, bank_idx - 1] = (
                f_vec[mask] - banks_ends[bank_idx - 1]
            ) / (banks_ends[bank_idx] - banks_ends[bank_idx - 1])

            # Points in the second half of the triangle
            mask = np.logical_and(
                f_vec >= banks_ends[bank_idx], f_vec <= banks_ends[bank_idx + 1]
            )
            filterbank[mask, bank_idx - 1] = (
                banks_ends[bank_idx + 1] - f_vec[mask]
            ) / (banks_ends[bank_idx + 1] - banks_ends[bank_idx])

        # Scale and normalize, so that all the triangles do not have same height
        # and the gain gets adjusted appropriately.
        temp = filterbank.sum(axis=0)
        non_zero_mask = temp > 0
        filterbank[:, non_zero_mask] /= np.expand_dims(temp[non_zero_mask], 0)

        # Save to cache
        CACHE[fname] = (filterbank, banks_ends[1:-1])

    return CACHE[fname][0], CACHE[fname][1]


def spectrogram(sig, rate):

    global IM_SIZE
    global WIN_LEN
    global FMIN
    global FMAX
    global MAGNITUDE_SCALE
    global BANDPASS

    # Compute overlap
    hop_len = int(len(sig) / (IM_SIZE[1] - 1))
    win_overlap = WIN_LEN - hop_len + 2

    # Adjusting N_FFT
    n_fft = WIN_LEN

    # Applying Bandpass Filter
    sig = applyBandpassFilter(sig, rate)

    # Compute spectrogram
    f, t, spec = scipy.signal.spectrogram(
        sig,
        fs=rate,
        window=scipy.signal.windows.hann(WIN_LEN),
        nperseg=WIN_LEN,
        noverlap=win_overlap,
        nfft=n_fft,
        detrend=False,
        mode="magnitude",
    )

    # Scaling frequency (melspec)
    # Determine the indices of where to clip the spec
    valid_f_idx_start = f.searchsorted(FMIN, side="left")
    valid_f_idx_end = f.searchsorted(FMAX, side="right") - 1

    # Get mel filter banks
    mel_filterbank, mel_f = get_mel_filterbanks(IM_SIZE[0], f, dtype=spec.dtype)

    # Clip to non-zero range so that unnecessary
    # multiplications can be avoided
    mel_filterbank = mel_filterbank[valid_f_idx_start : (valid_f_idx_end + 1), :]

    # Clip the spec representation and apply the mel filterbank.
    # Due to the nature of np.dot(), the spec needs to be transposed prior,
    # and reverted after.
    spec = np.transpose(spec[valid_f_idx_start : (valid_f_idx_end + 1), :], [1, 0])
    spec = np.dot(spec, mel_filterbank)
    spec = np.transpose(spec, [1, 0])

    # Convert magnitudes using nonlinearity as proposed by Schlüter, 2018
    a = -1.2  # Higher values yield better noise suppression
    s = 1.0 / (1.0 + np.exp(-a))
    spec = spec ** s

    # Flip spectrum vertically (only for better visualization, low freq. at bottom)
    spec = spec[::-1, ...]

    # Trim to desired shape if too large
    spec = spec[: IM_SIZE[0], : IM_SIZE[1]]

    # Normalize values between 0 and 1
    spec -= spec.min()

    if not spec.max() == 0:
        spec /= spec.max()
    else:
        spec = np.clip(spec, 0, 1)

    return spec


def splitSignal(sig, rate):
    """
    Splits a signal into chunks with possible overlap.
    """

    global LENGTH
    global OVERLAP
    global MINLEN

    # Split signal with overlap
    sig_splits = []
    sig_chunk_size = int((LENGTH - OVERLAP) * rate)
    sig_length = len(sig)

    for i in range(0, sig_length, sig_chunk_size):
        split = sig[i : i + int(LENGTH * rate)]

        # If we cannot retrieve anymore splits (end of signal), break
        if len(split) < int(MINLEN * rate):
            break

        # If signal chunk is long enough to be considered but not as long as previous chunks,
        # pad chunk with noise.
        if len(split) < int(rate * LENGTH):
            split = np.hstack(
                (split, noise(split, (int(rate * LENGTH) - len(split)), 0.5))
            )

        sig_splits.append(split)

    return sig_splits


def specsFromSignal(sig, rate):
    """
    Extracts spectrograms from a signal.
    """

    # Split signal into consecutive chunks with overlap
    sig_splits = splitSignal(sig, rate)

    for sig in sig_splits:
        # Get spectrogram for each signal chunk
        spec = spectrogram(sig, rate)
        yield spec


def specsFromTimestamps(audio, timestamps):
    """
    Produces spectrograms corresponding to audio clips that will be used to produce
    feature vectors for clustering later on.

    Parameters
    ----------
    audio : AudioSegment object
        The AudioSegment taken from previous stages that represents the raw
        audio clip being passed in as input.

    timestamps: list of tuples of int
        Contains timestamps described as a pair (start, end) where bird calls were
        detected.

    Yields
    ------
    spec : numpy array of shape (64, 384)
        Spectrograms that are to be produced in order to run the BirdNET model.
    """
    audio_tot_len = audio.duration_seconds
    for timestamp in timestamps:
        sig, rate = getAudioSlice(audio, audio_tot_len, timestamp[0], timestamp[1])

        # Yield all specs for file
        for spec in specsFromSignal(sig, rate):
            yield timestamp, spec
