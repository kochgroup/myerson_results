def infer_dims(data, data_type):
    """Automatically set the dimensions for the neural network.

    Args:
        data (any): Single data instance from which to autodetect the dimensions.
        data_type (str): Whether graph, tabular, image or other data.

    Returns:
        Tuple[int]: dim_in, dim_out, dim_hidden
    """
    if data_type == 'graph':
        dim_in = data.x.shape[1]
        if len(data.y.shape) == 2:
            dim_out = data.y.shape[1]
        elif len(data.y.shape) == 1:
            dim_out = data.y.shape[0]
        else:
            raise ValueError(f"Unexpected shape of dataset labels {data.y.shape}, could not assign out_dim.")
    else:
        raise NotImplementedError(f"Data type {data_type} not implemented.")
    return dim_in, dim_out 