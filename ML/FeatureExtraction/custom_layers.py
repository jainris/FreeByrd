import tensorflow as tf
from tensorflow import keras as k
from tensorflow.keras import layers as l


class SimpleSpecLayer(l.Layer):
    def __init__(
        self,
        sample_rate=48000,
        spec_shape=(257, 384),
        frame_step=374,
        frame_length=512,
        data_format="channels_last",
        **kwargs
    ):
        super(SimpleSpecLayer, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.spec_shape = spec_shape
        self.data_format = data_format
        self.frame_step = frame_step
        self.frame_length = frame_length

    def build(self, input_shape):
        self.mag_scale = self.add_weight(
            name="magnitude_scaling",
            initializer=k.initializers.Constant(value=1.0),
            trainable=True,
        )
        super(SimpleSpecLayer, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_last":
            return tf.TensorShape((None, self.spec_shape[0], self.spec_shape[1], 1))
        else:
            return tf.TensorShape((None, 1, self.spec_shape[0], self.spec_shape[1]))

    def call(self, inputs):

        # Perform STFT
        spec = tf.signal.stft(
            inputs,
            self.frame_length,
            self.frame_step,
            fft_length=self.frame_length,
            window_fn=tf.signal.hann_window,
            pad_end=False,
            name="stft",
        )

        # Cast from complex to float
        spec = tf.dtypes.cast(spec, tf.float32)

        # Convert to power spectrogram
        spec = tf.math.pow(spec, 2.0)

        # Convert magnitudes using nonlinearity
        spec = tf.math.pow(spec, 1.0 / (1.0 + tf.math.exp(self.mag_scale)))

        # Normalize values between 0 and 1
        spec = tf.math.divide(
            tf.math.subtract(spec, k.backend.min(spec, axis=[1, 2], keepdims=True)),
            k.backend.max(spec, axis=[1, 2], keepdims=True),
        )

        # Swap axes to fit input shape
        spec = tf.transpose(spec, [0, 2, 1])

        # Add channel axis
        if self.data_format == "channels_last":
            spec = tf.expand_dims(spec, -1)
        else:
            spec = tf.expand_dims(spec, 1)

        return spec

    def get_config(self):
        config = {
            "data_format": self.data_format,
            "sample_rate": self.sample_rate,
            "spec_shape": self.spec_shape,
            "frame_step": self.frame_step,
            "frame_length": self.frame_length,
        }
        base_config = super(SimpleSpecLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))