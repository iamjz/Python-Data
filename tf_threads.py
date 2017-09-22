def limit(tf, n):
    """ Limit parallelization level """
    tfconfig = tf.ConfigProto(
        intra_op_parallelism_threads=n, # thread-pool size for graph
        inter_op_parallelism_threads=2, # thread-pool size for ops
        allow_soft_placement=True, # divert to CPU when GPU is unavailable regardless of explicit device placement
        device_count = {
            'CPU': 1, # enable 1 CPU device
            'GPU': 0, # disable any GPU device
        })
    return tfconfig
