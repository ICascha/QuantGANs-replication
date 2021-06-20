from tensorflow.keras.layers import PReLU, Conv1D, Add, Input, Cropping2D, Conv2D, Concatenate, Lambda, Activation
from tensorflow.keras.models import Model
from tensorflow.compat.v1.keras.layers import BatchNormalization
from tensorflow_addons.layers import SpectralNormalization

def add_temporal_block(prev_layer, skip_layer, kernel_size, dilation, fixed_filters, moving_filters, n_series, rfs, block_size, use_batchNorm, cropping):
    """Creates a temporal block.
    Args:
        prev_layer (tensorflow.keras.layers.Layer): previous layer to attach to on standard path.
        skip_layer (tensorflow.keras.layers.Layer): previous layer to attach to on the skip path. Use None for intiation.
        kernel_size (int): kernel size along temporal axis of convolution layers within the temporal block.
        dilation (int): dilation of convolution layers along temporal axis within the temporal block.
        n_filters (int): Number of channels to use within the temporal block.
        n_series (int): Number of timeseries to model.
        rfs (int): Receptive field size of the network.
        block_size (int): Number of convolution layers to use within a temporal block.
        use_batchNorm (bool): Whether to use batch normalization (with renormalization).
    Returns:
        tuple of tensorflow.keras.layers.Layer: Output layers belonging to (normal path, skip path).
    """
    # Identity mapping so that we hold a valid reference to prev_layer
    block = Lambda(lambda x: x)(prev_layer)

    for _ in range(block_size):
        convs = []
        for _ in range(n_series):
            prev_block= Lambda(lambda x: x)(block)
            convs.append(SpectralNormalization(Conv2D(fixed_filters, (n_series, kernel_size), dilation_rate=(1, dilation)))(block))

        
        if len(convs) > 1:
		block = Concatenate(axis=1)(convs) 
	else:
		block = convs[0]
        if moving_filters:
            block = Concatenate(axis=-1)([block, Conv2D(moving_filters, (1, kernel_size), dilation_rate=(1, dilation))(prev_block)])
        if use_batchNorm:
            block = BatchNormalization(axis=3, momentum=.9, epsilon=1e-5, renorm=True, renorm_momentum=.9)(block)

        block = PReLU(shared_axes=[2, 3])(block)
        
    # As layer output gets smaller, we need to crop less before putting output
    # on the skip path. We cannot infer this directly as tensor shapes may be variable.
    drop_left = block_size * (kernel_size - 1) * dilation
    cropping += drop_left

    if skip_layer is None:
        prev_layer = Conv2D(fixed_filters + moving_filters, 1)(prev_layer)
    # add residual connections
    out = Add()([Cropping2D(cropping=((0,0), (drop_left, 0)))(prev_layer), block])
    # crop from left side for skip path
    skip_out = Cropping2D(cropping=((0,0), (rfs-1-cropping, 0)))(out)
    # add current output with 1x1 conv to skip path
    if skip_layer is not None:
        skip_out = Add()([skip_layer, SpectralNormalization(Conv2D(fixed_filters + moving_filters, 1))(skip_out)])
    else:
        skip_out = SpectralNormalization(Conv2D(fixed_filters + moving_filters, 1))(skip_out)

    return PReLU(shared_axes=[2, 3])(out), skip_out, cropping
	
def make_TCN(dilations, fixed_filters, moving_filters, use_batchNorm, one_series_output, sigmoid, input_dim, block_size=2):
    """Creates a causal temporal convolutional network with skip connections.
       This network uses 2D convolutions in order to model multiple timeseries co-dependency.
    Args:
        dilations (list, tuple): Ordered number of dilations to use for the temporal blocks.
        fixed_filters (int): Number of channels in the hidden layers corresponding fixed over series axis.
        moving_filters (int): Number of channels in the hidden layers moving over series axis.
        use_batchNorm (bool): Whether to use batch normalization in the temporal blocks. Includes batch Renormalization.
        one_series_output (bool): Whether to collapse the dimension of the series axis to 1 using an additional convolution layer.
        sigmoid (bool): Whether to append the sigmoid activation function at the output of the network.
        input_dim (list, tuple): Input dimension of the shape (number of timeseries, timesteps, number of features). Timesteps may be None for variable length timeseries. 
        block_size (int): How many convolution layers to use within a temporal block. Defaults to 2.
    Returns:
        tensorflow.keras.models.Model: a non-compiled keras model.
    """    
    rfs = receptive_field_size(dilations, block_size)
    n_series = input_dim[0]

    input_layer = Input(shape=input_dim)
    cropping = 0
    prev_layer, skip_layer, _ = add_temporal_block(input_layer, None, 1, 1, fixed_filters, moving_filters, n_series, rfs, block_size, use_batchNorm, cropping)
                
    for dilation in dilations:
        prev_layer, skip_layer, cropping = add_temporal_block(prev_layer, skip_layer, 2, dilation, fixed_filters, moving_filters, n_series, rfs, block_size, use_batchNorm, cropping)

    output_layer = PReLU(shared_axes=[2, 3])(skip_layer)
    output_layer = SpectralNormalization(Conv2D(fixed_filters + moving_filters, kernel_size=1))(output_layer)
    output_layer = PReLU(shared_axes=[2, 3])(output_layer)
    output_layer = SpectralNormalization(Conv2D(1, kernel_size=1))(output_layer)

    if one_series_output:
        output_layer = SpectralNormalization(Conv2D(1, (n_series, 1)))(output_layer)

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
