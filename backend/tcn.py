from tensorflow.keras.layers import PReLU, Conv1D, Add, Input, Activation, Cropping2D, ZeroPadding2D, Conv2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow_addons.layers import SpectralNormalization, WeightNormalization
from tensorflow.compat.v1.keras.layers import BatchNormalization

def add_temporal_block(prev_layer, skip_layer, kernel_size, dilation, n_filters, n_series, rfs, block_size, use_batchNorm, cropping):
    """Creates a temporal block.

    Args:
        prev_layer (tensorflow.keras.layers.Layer): previous layer to attach to on standard path.
        skip_layer (tensorflow.keras.layers.Layer): previous layer to attach to on the skip path.
        kernel_size (int): kernel size of convolution layers within the temporal block.
        dilation (int): dilation of convolution layers within the temporal block.
        n_filters (int): Number of channels to use within the temporal block.
        n_series (int): Number of timeseries to model.
        rfs (int): Receptive field size of the network.
        block_size (int): Number of convolution layers to use within a temporal block.
        use_batchNorm (bool): Whether to use batch normalization (with renormalization).

    Returns:
        tuple of tensorflow.keras.layers.Layer: Output layers belonging to (normal path, skip path).
    """
    convs = []
    for _ in range(n_series):
        convs.append(SpectralNormalization(Conv2D(n_filters, (n_series, kernel_size), dilation_rate=(1, dilation)))(prev_layer))
    if len(convs) == 1:
        block = convs[0]
    else:
        block = Concatenate(axis=1)(convs)
    # Pad cusually
    block = ZeroPadding2D(padding=((0, 0), (dilation//2, 0)))(block)
    block = PReLU(shared_axes=[1, 2, 3])(block)
    if use_batchNorm:
        block = BatchNormalization(axis=3, momentum=.9, epsilon=1e-5, renorm=True, renorm_momentum=.9)(block)

    for _ in range(block_size - 1):
        convs = []
        for _ in range(n_series):
            convs.append(SpectralNormalization(Conv2D(n_filters, (n_series, kernel_size), dilation_rate=(1, dilation)))(block))
        if len(convs) == 1:
            block = convs[0]
        else:
            block = Concatenate(axis=1)(convs)
        block = ZeroPadding2D(padding=((0,0), (dilation//2, 0)))(block)
        block = PReLU(shared_axes=[1, 2, 3])(block)
        if use_batchNorm:
            block = BatchNormalization(axis=3, momentum=.9, epsilon=1e-5, renorm=True, renorm_momentum=.9)(block)
        
    if kernel_size != 1:
        crop_temp = cropping
        cropping = drop_left = block_size * (kernel_size - 1) * dilation
        drop_left -= crop_temp
    else:
        # upsample channel axis
        prev_layer = Conv2D(n_filters, 1)(prev_layer)
        # add residual connections
        out = Add()([prev_layer,block])
        # crop from left side for skip path
        skip_out = Cropping2D(cropping=((0,0), (rfs-1, 0)))(out)
        # add 1x1 convolution to skip path
        skip_out = SpectralNormalization(Conv2D(n_filters, 1))(skip_out)

        return PReLU(shared_axes=[1, 2, 3])(out), skip_out, cropping
    # add residual connections
    out = Add()([Cropping2D(cropping=((0,0), (drop_left, 0)))(prev_layer),block])
    # crop from left side for skip path
    skip_out = Cropping2D(cropping=((0,0), (rfs-1-cropping, 0)))(out)
    # add current output with 1x1 conv to skip path
    skip_out = Add()([skip_layer, SpectralNormalization(Conv2D(n_filters, 1))(skip_out)])

    return PReLU(shared_axes=[1,2, 3])(out), skip_out, cropping


def make_TCN(dilations, n_filters, use_batchNorm, one_series_output, sigmoid, input_dim, block_size=2):
    """Creates a causal temporal convolutional network with skip connections.
       This network uses 2D convolutions in order to model multiple timeseries co-dependency.

    Args:
        dilations (list, tuple): Ordered number of dilations to use for the temporal blocks.
        n_filters (int): Number of channels in the hidden layers.
        use_batchNorm (int): Whether to use batch normalization in the temporal blocks. Includes batch Renormalization.
        sigmoid (bool): Whether to append the sigmoid activation function at the output of the network.
        input_dim (list, tuple): Input dimension of the shape (number of timeseries, timesteps, number of features). Timesteps may be None for variable length timeseries. 
        block_size (int): How many convolution layers to use within a temporal block. Defaults to 2.

    Returns:
        tensorflow.keras.models.Model: a non-compiled keras model.
    """    
    rfs = receptive_field_size(dilations, block_size)
    n_series = input_dim[0]

    input_layer = Input(shape=input_dim)

    prev_layer, skip_layer, _ = add_temporal_block(input_layer, None, 1, 1, n_filters, n_series, rfs, block_size, use_batchNorm, None)
            
    cropping = 0
    for dilation in dilations:
        prev_layer, skip_layer, cropping = add_temporal_block(prev_layer, skip_layer, 2, dilation, n_filters, n_series, rfs, block_size, use_batchNorm, cropping)

    output_layer = PReLU(shared_axes=[1, 2, 3])(skip_layer)
    output_layer = SpectralNormalization(Conv2D(n_filters, kernel_size=1))(output_layer)
    output_layer = PReLU(shared_axes=[1, 2, 3])(output_layer)
    output_layer = SpectralNormalization(Conv2D(1, kernel_size=1))(output_layer)

    if one_series_output:
        output_layer = Conv2D(1, (n_series, 1))(output_layer)

    if sigmoid:
        output_layer = Activation('sigmoid')(output_layer)

    return Model(input_layer, output_layer)

def receptive_field_size(dilations, block_size):
    """Non-exhaustive computation of receptive field size.

    Args:
        dilations (list, tuple): Ordered number of dilations of the network.
        block_size (int): Number of convolution layers in each temporal block of the network.

    Returns:
        int: the receptive field size.
    """    
    return 1 + block_size * sum(dilations)