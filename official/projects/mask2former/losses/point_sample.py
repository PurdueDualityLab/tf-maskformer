import tensorflow as tf


def safe_get(input, n, x, y, c):
    """
    Provide zero padding for getting values from out of bounds indices from the 'input' matrix
    """
    # input shape = [N, H, W, C]
    H = input.shape[1]
    W = input.shape[2]

    return input[n, y, x, c] if (x >= 0 and x < W and y >= 0 and y < H) else 0


def point_sample(input, point_coords, align_corners=False, **kwargs):
    """
    A TensorFlow implementation of point_sample from detectron. Sample points
    using bilinear interpolation from `point_coords` which is assumed to be a
    [0, 1] x [0, 1] square. Default mode for align_corners is False
    Args:
        input (Tensor): A tensor of shape (N, H, W, C) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains [0, 1] x [0, 1]
        normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, P, C) containing features for points
        in `point_coords`. The features are obtained via bilinear interpolation from
        `input`.
    """

    # assert correct dimensions
    assert len(input.shape) == 4
    assert len(point_coords.shape) == 3

    N = input.shape[0]
    H = input.shape[1]
    W = input.shape[2]
    C = input.shape[3]

    P = point_coords.shape[1]

    # changing x,y range from [0, 1] to [-1, 1]
    point_coords = 2 * point_coords - 1

    output = tf.zeros([N, P, C])

    for n in range(N):
        for p in range(P):
            x, y = point_coords[n, p, :]

            if align_corners:
                # Unnormalize coords from [-1, 1] to [0, H - 1] & [0, W - 1]
                x = ((x + 1) / 2) * (W - 1)
                y = ((y + 1) / 2) * (H - 1)
            else:
                # Unnormalize coords from [-1, 1] to [-0.5, H - 0.5] & [-0.5, W - 0.5]
                x = ((x + 1) * W - 1) / 2
                y = ((y + 1) * H - 1) / 2

            x1, y1 = int(tf.floor(x)), int(tf.floor(y))
            x2, y2 = x1 + 1, y1 + 1

            for c in range(C):
                nw_val = safe_get(input, n, x1, y1, c)
                sw_val = safe_get(input, n, x1, y2, c)
                ne_val = safe_get(input, n, x2, y1, c)
                se_val = safe_get(input, n, x2, y2, c)

                R1 = ((x2 - x) / (x2 - x1)) * nw_val + ((x - x1) / (x2 - x1)) * ne_val
                R2 = ((x2 - x) / (x2 - x1)) * sw_val + ((x - x1) / (x2 - x1)) * se_val
                sampled_point = ((y2 - y) / (y2 - y1)) * R1 + (
                    (y - y1) / (y2 - y1)
                ) * R2

                # might be inefficiently if tensor_scatter_nd_update is creating a new matrix for each index update
                output = tf.tensor_scatter_nd_update(
                    output, indices=[[n, p, c]], updates=[sampled_point]
                )

    return output
