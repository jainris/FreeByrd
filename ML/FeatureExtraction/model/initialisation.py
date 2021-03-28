# Adapted from https://github.com/kahst/BirdNET
import pickle

import theano
import theano.tensor as t

from lasagne import layers as l
from lasagne import nonlinearities as nl

FILTERS = [8, 16, 32, 64, 128]


class Model:
    def __init__(self, model):
        self.model = model
        output_layer = l.get_output(l.get_all_layers(model)[-1], deterministic=True)
        self.function = theano.function(
            [l.get_all_layers(model)[0].input_var],
            output_layer,
            allow_input_downcast=True,
        )

    def run_model(self, spect):
        return self.function(spect)


def load_model(path):
    """

    This function builds up a lasagne network, who's architecture is a modified
    birdNet arichitecture, from which we can extract features. For the original
    network architecture, see the original paper:
    https://www.sciencedirect.com/science/article/pii/S1574954121000273

    Parameters
    ----------
    path: string
        This represents the location of the downloaded birdNet model file,
        from which we extract the relevant parameters

    Returns
    -------
    model: Model
        Returns a model object, which holds the relevant information for the

    """

    with open(path, "rb") as f:
        try:
            data = pickle.load(f, encoding="latin1")
        except:
            data = pickle.load(f)

    params = data["params"][
        :169
    ]  # we don't use the classification layers, so slice the paramaters from there
    net = get_network()
    l.set_all_param_values(net, params)

    model = Model(net)
    return model


# This sets up the model architecure to match that of BirdNET
def get_network():
    """

    This function builds up a lasagne network, who's architecture is a modified
    birdNet arichitecture, from which we can extract features. For the original
    network architecture, see https://www.sciencedirect.com/science/article/pii/S1574954121000273

    Parameters
    ----------
    none

    Returns
    -------
    model: lasagne_network
        The relevant network

    """
    model = l.InputLayer((1, 1, 64, 384))  # Input spectograms, should be 1 x 64 x 384

    # First "preprocessing" convolutional layer, with batch normalisation, RELU activation and Max Pooling
    model = l.batch_norm(
        l.Conv2DLayer(
            model,
            num_filters=32,
            filter_size=(5, 5),
            pad="same",
            nonlinearity=nl.rectify,
        )
    )
    model = l.MaxPool2DLayer(model, pool_size=(1, 2))

    # We now have 4 residual blocks, each one having one downsampling layer and 2 "normal" residual layers
    for i in range(1, 5):
        model = get_residual(
            model, filters=FILTERS[i] * 4, downsampling=True, blockNum=i
        )
        for j in range(0, 2):
            model = get_residual(model, filters=int(FILTERS[i] * 4), blockNum=i)

    model = l.batch_norm(model)
    model = l.NonlinearityLayer(model, nonlinearity=nl.rectify)

    return model


# This is a helper function, called to create our residual blocks
# @params:


def get_residual(input, filters, downsampling=False, blockNum=1):
    """

        This builds the residual blocks within the neural network

        Parameters
        ----------
        input: lasagne_network
            the model on which to add the layers

        filters: int
            the size of filters for the residual block

        downsampling: bool, optional
            whether or not this layer performs downsampling or not
            (only the first layer does).


        Returns
        -------
        model: lasagne_network
            The relevant network

        """
    # We apply RELU to inputs, apart from the very first downsampling layer as we are taking already RELU'd input
    if blockNum == 1 and downsampling:
        net_pre = input
    else:
        net_pre = l.NonlinearityLayer(input, nonlinearity=nl.rectify)

    # We start off with a bottleneck convolution on the input (although interestingly, no reduction in dimensonality
    # is actually done in the model, unsure why)
    if downsampling:
        net_pre = l.batch_norm(
            l.Conv2DLayer(
                net_pre,
                filter_size=1,
                num_filters=l.get_output_shape(net_pre)[1],
                pad="same",
                stride=1,
                nonlinearity=nl.rectify,
            )
        )

    # We do our first convolution
    out = l.batch_norm(
        l.Conv2DLayer(
            net_pre,
            num_filters=l.get_output_shape(net_pre)[1],
            filter_size=(3, 3),
            pad="same",
            stride=1,
            nonlinearity=nl.rectify,
        )
    )

    # This is our downsampling, which we only do on the first residual layer in our residual block
    if downsampling:
        out = l.MaxPool2DLayer(out, pool_size=(2, 2))

    # A dropout layer- this is not used by us as we don't train the network, but we wish to have the network to have
    # the same architecture as the one used by BirdNET to make setting parameters easier
    out = l.DropoutLayer(out)

    # We do a second convolution here
    out = l.batch_norm(
        l.Conv2DLayer(
            out,
            num_filters=filters,
            filter_size=(3, 3),
            pad="same",
            stride=1,
            nonlinearity=None,
        )
    )

    # if we have Downsampled, need to change our residual net shortcut a bit
    if downsampling:
        # We do some average pooling, this matches our downsampling correctly and gives us a same sized tensor for the shortcut
        shortcut = l.Pool2DLayer(
            input, stride=2, pool_size=(2, 2), mode="average_exc_pad"
        )

        shortcut = l.batch_norm(
            l.Conv2DLayer(
                shortcut,
                num_filters=filters,
                filter_size=1,
                pad="same",
                stride=1,
                nonlinearity=None,
            )
        )
    else:
        shortcut = input

    out = l.ElemwiseSumLayer([out, shortcut])

    return out