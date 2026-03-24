import math
import torch
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def generate_input_representation(
    events, event_representation, shape, nr_temporal_bins=5, separate_pol=True
):
    """
    Events: N x 4, where cols are x, y, t, polarity, and polarity is in {-1, 1}. x and y correspond to image
    coordinates u and v.
    """

    if isinstance(event_representation, str):
        event_representation = [event_representation]

    def concatenate_arrays(*arrays, axis=0):
        arrays = [np.array(a) for a in arrays if a is not None]
        if len(arrays) == 0:
            return None
        elif len(arrays) == 1:
            return arrays[0]
        else:
            return np.concatenate(arrays, axis=axis)

    event_historgram, event_voxel, event_mean_var = None, None, None
    for rep_method in event_representation:
        if rep_method == "historgram":
            event_historgram = generate_event_histogram(
                events, shape, mean=False, variance=False
            )
        elif rep_method == "voxel_grid":
            event_voxel = generate_voxel_grid(
                events, shape, nr_temporal_bins, separate_pol
            )
        elif rep_method == "mean_variance":
            event_mean_var = generate_event_histogram(
                events, shape, mean=True, variance=True
            )

    event_rep = concatenate_arrays(event_voxel, event_historgram, event_mean_var)

    return event_rep


def generate_event_histogram(events, shape, mean=False, variance=False):
    """
    Events: N x 4, where cols are x, y, t, polarity, and polarity is in {-1, 1}. x and y correspond to image
    coordinates u and v.
    """
    height, width = shape
    x, y, t, p = events.T
    x = x.astype(int)
    y = y.astype(int)
    p[p == 0] = -1  # polarity should be +1 / -1
    img_pos = np.zeros((height * width,), dtype="float32")
    img_neg = np.zeros((height * width,), dtype="float32")

    mask = (x < height) & (x > 0) & (y < width) & (y > 0) & (t > 0)
    x = x[mask]
    y = y[mask]
    t = t[mask]
    p = p[mask]

    pos_x = x[p == 1]
    pos_y = y[p == 1]
    neg_x = x[p == -1]
    neg_y = y[p == -1]

    np.add.at(img_pos, pos_x + width * pos_y, 1)
    np.add.at(img_neg, neg_x + width * neg_y, 1)
    histogram = np.stack([img_neg, img_pos], 0).reshape((2, height, width))

    if mean:
        norm_t = (t - t[0]) / (t[-1] + 1e-6)
        # Mean
        mean_pos = np.zeros((height * width,), dtype="float32")
        mean_neg = np.zeros((height * width,), dtype="float32")

        mean_pos[pos_x + width * pos_y] += norm_t[p == 1]
        mean_neg[neg_x + width * neg_y] += norm_t[p == -1]

        mean_pos = mean_pos / (img_pos + 1e-6)
        mean_neg = mean_neg / (img_neg + 1e-6)
        mean = np.stack([mean_pos, mean_neg], 0).reshape((2, height, width))

        if variance:
            # Variance
            var_pos = np.zeros((height * width,), dtype="float32")
            var_neg = np.zeros((height * width,), dtype="float32")
            var_pos[pos_x + width * pos_y] += (
                norm_t[p == 1] - mean_pos[pos_x + width * pos_y]
            ) ** 2
            var_neg[neg_x + width * neg_y] += (
                norm_t[p == -1] - var_neg[neg_x + width * neg_y]
            ) ** 2
            var_pos = np.sqrt(var_pos / (img_pos - 1 + 1e-6))
            var_neg = np.sqrt(var_neg / (img_neg - 1 + 1e-6))
            var = np.stack([var_pos, var_neg], 0).reshape((2, height, width))
            event_rep = np.concatenate((mean, var), axis=0)
        else:
            event_rep = mean
    else:
        event_rep = histogram

    return event_rep


def normalize_voxel_grid(events):
    """Normalize event voxel grids"""
    nonzero_ev = events != 0
    num_nonzeros = nonzero_ev.sum()
    if num_nonzeros > 0:
        # compute mean and stddev of the **nonzero** elements of the event tensor
        # we do not use PyTorch's default mean() and std() functions since it's faster
        # to compute it by hand than applying those funcs to a masked array
        mean = events.sum() / num_nonzeros
        stddev = torch.sqrt((events**2).sum() / num_nonzeros - mean**2)
        mask = nonzero_ev.float()
        events = mask * (events - mean) / stddev

    return events


def voxel_normalization(voxel):
    """
    normalize the voxel same as https://arxiv.org/abs/1912.01584 Section 3.1
    Params:
        voxel: torch.Tensor, shape is [num_bins, H, W]

    return:
        normalized voxel
    """
    # check if voxel all element is 0
    a, b, c = voxel.shape
    tmp = torch.zeros(a, b, c)
    if torch.equal(voxel, tmp):
        return voxel
    abs_voxel, _ = torch.sort(torch.abs(voxel).view(-1, 1).squeeze(1))
    # print("abs_voxel.shape: ", abs_voxel.shape)
    first_non_zero_idx = torch.nonzero(abs_voxel)[0].item()
    non_zero_voxel = abs_voxel[first_non_zero_idx:]
    norm_idx = math.floor(non_zero_voxel.shape[0] * 0.98)

    ones = torch.ones_like(voxel)

    # squeeze_voxel, indices = torch.sort(voxel.view(-1, 1).squeeze(1))
    normed_voxel = torch.where(
        torch.abs(voxel) < non_zero_voxel[norm_idx],
        voxel / non_zero_voxel[norm_idx],
        voxel,
    )
    normed_voxel = torch.where(
        normed_voxel >= non_zero_voxel[norm_idx], ones, normed_voxel
    )
    normed_voxel = torch.where(
        normed_voxel <= -non_zero_voxel[norm_idx], -ones, normed_voxel
    )

    return normed_voxel


def generate_voxel_grid(events, shape, nr_temporal_bins, separate_pol=False):
    """
    Build a voxel grid with bilinear interpolation in the time domain from a set of events.
    :param events: a [N x 4] NumPy array containing one event per row in the form: [x, y, timestamp, polarity]
    :param nr_temporal_bins: number of bins in the temporal axis of the voxel grid
    :param shape: dimensions of the voxel grid
    """
    height, width = shape
    assert events.shape[1] == 4
    assert nr_temporal_bins > 0
    assert width > 0
    assert height > 0

    voxel_grid_positive = np.zeros(
        (nr_temporal_bins, height, width), np.float32
    ).ravel()
    voxel_grid_negative = np.zeros(
        (nr_temporal_bins, height, width), np.float32
    ).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 2]
    first_stamp = events[0, 2]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0

    xs = events[:, 0].astype(int)
    ys = events[:, 1].astype(int)
    ts = (nr_temporal_bins - 1) * (events[:, 2] - first_stamp) / deltaT

    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = np.abs(pols) * (1.0 - dts)
    vals_right = np.abs(pols) * dts
    pos_events_indices = pols == 1

    # Positive Voxels Grid
    valid_indices_pos = np.logical_and(tis < nr_temporal_bins, pos_events_indices)
    valid_pos = (
        (xs < width)
        & (xs >= 0)
        & (ys < height)
        & (ys >= 0)
        & (ts >= 0)
        & (ts < nr_temporal_bins)
    )
    valid_indices_pos = np.logical_and(valid_indices_pos, valid_pos)

    np.add.at(
        voxel_grid_positive,
        xs[valid_indices_pos]
        + ys[valid_indices_pos] * width
        + tis[valid_indices_pos] * width * height,
        vals_left[valid_indices_pos],
    )

    valid_indices_pos = np.logical_and((tis + 1) < nr_temporal_bins, pos_events_indices)
    valid_indices_pos = np.logical_and(valid_indices_pos, valid_pos)
    np.add.at(
        voxel_grid_positive,
        xs[valid_indices_pos]
        + ys[valid_indices_pos] * width
        + (tis[valid_indices_pos] + 1) * width * height,
        vals_right[valid_indices_pos],
    )

    # Negative Voxels Grid
    valid_indices_neg = np.logical_and(tis < nr_temporal_bins, ~pos_events_indices)
    valid_indices_neg = np.logical_and(valid_indices_neg, valid_pos)

    np.add.at(
        voxel_grid_negative,
        xs[valid_indices_neg]
        + ys[valid_indices_neg] * width
        + tis[valid_indices_neg] * width * height,
        vals_left[valid_indices_neg],
    )

    valid_indices_neg = np.logical_and(
        (tis + 1) < nr_temporal_bins, ~pos_events_indices
    )
    valid_indices_neg = np.logical_and(valid_indices_neg, valid_pos)
    np.add.at(
        voxel_grid_negative,
        xs[valid_indices_neg]
        + ys[valid_indices_neg] * width
        + (tis[valid_indices_neg] + 1) * width * height,
        vals_right[valid_indices_neg],
    )

    voxel_grid_positive = np.reshape(
        voxel_grid_positive, (nr_temporal_bins, height, width)
    )
    voxel_grid_negative = np.reshape(
        voxel_grid_negative, (nr_temporal_bins, height, width)
    )

    if separate_pol:
        return np.concatenate([voxel_grid_positive, voxel_grid_negative], axis=0)

    voxel_grid = voxel_grid_positive - voxel_grid_negative

    return voxel_grid
