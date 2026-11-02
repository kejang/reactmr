import numpy as np
from scipy.fftpack import dct, idct
from sklearn.cluster import KMeans
import cupy as cp


def from_device_array(d_x, *, gpu_id, stream):

    xp = cp.get_array_module(d_x)

    if xp == np:
        x = d_x.copy()  # already on host

    else:
        with cp.cuda.Device(gpu_id), stream as stream:
            block_events = cp.cuda.Event(block=True)

            x = np.empty(d_x.shape, dtype=d_x.dtype)
            d_x.get(out=x, stream=stream)

            block_events.record(stream)
            block_events.synchronize()
            block_events = None

    return x


def to_device_array(
    x,
    *,
    gpu_id,
    stream,
    dtype_real="float32",
    dtype_complex="complex64",
    shape=None,
    ravel=False,
    create_copy=True,
):

    xp = cp.get_array_module(x)

    if xp.iscomplexobj(x):
        dtype = cp.dtype(dtype_complex)
    else:
        dtype = cp.dtype(dtype_real)

    d_x = None

    if xp == np:  # numpy -> cupy
        with cp.cuda.Device(gpu_id), stream as stream_:
            block_events = cp.cuda.Event(block=True)
            if shape is None:
                if ravel:
                    d_x = cp.empty((x.size,), dtype=dtype)
                    d_x.set(x.astype(dtype).ravel(), stream=stream_)
                    block_events.record(stream_)
                else:
                    d_x = cp.empty(x.shape, dtype=dtype)
                    d_x.set(x.astype(dtype), stream=stream_)
                    block_events.record(stream_)
            else:
                temp_x = x.astype(dtype).reshape(shape)
                d_x = cp.empty(temp_x.shape, dtype=dtype)
                d_x.set(temp_x, stream=stream_)
                block_events.record(stream_)

            block_events.synchronize()
            block_events = None

    else:  # already on device
        with cp.cuda.Device(gpu_id), stream as _:
            if shape is None:
                if ravel:
                    if create_copy:
                        d_x = cp.array(x.ravel(), dtype=dtype)
                    else:
                        d_x = x.ravel().astype(dtype)
                else:
                    if create_copy:
                        d_x = cp.array(x, dtype=dtype)
                    else:
                        d_x = x.astype(dtype)
            else:
                if create_copy:
                    d_x = cp.array(x.reshape(shape), dtype=dtype)
                else:
                    d_x = x.reshape(shape).astype(dtype)

    return d_x


def get_roi_slice(mask, *, margin=0, tol=1e-9):
    """Return slices enclosing mask values with magnitude greater than ``tol``.

    ``margin`` adds voxels on each side, clipped to the image boundaries.
    Raise ValueError if no mask values exceed the threshold.
    """
    # Build mask_binary based on tolerance
    mask_binary = np.abs(mask) > tol

    # Find indices of mask_binary
    idx = np.where(mask_binary)
    if idx[0].size == 0:
        raise ValueError(f"No elements found with abs(value) > {tol}.")

    # Determine bounding box extents per dimension
    mins = np.array([arr.min() for arr in idx])
    maxs = np.array([arr.max() for arr in idx])

    # Apply margin and clamp to mask boundaries
    shape = np.array(mask.shape)
    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, shape - 1)

    # Construct slices
    slices = []
    for mn, mx in zip(mins, maxs):
        slices.append(slice(int(mn), int(mx) + 1))
    slices = tuple(slices)

    return slices


def shift_roi_slice(roi_slice, shift, shape):
    """Return ROI slices shifted by integer offsets in array-axis order.

    ``roi_slice``, ``shift``, and ``shape`` must have matching lengths.
    Clip to ``shape`` boundaries; raise ValueError if the crop becomes empty.
    """
    if len(roi_slice) != len(shift) or len(shape) != len(shift):
        raise ValueError("roi_slice, shift, and shape must have same length.")

    new_slices = []
    for sl, d, n in zip(roi_slice, shift, shape):
        # Interpret None as full range
        start = 0 if sl.start is None else sl.start
        stop = n if sl.stop is None else sl.stop  # stop is exclusive

        start = start + d
        stop = stop + d

        # Clamp to [0, n]
        start = max(0, min(start, n))
        stop = max(0, min(stop, n))

        if start >= stop:
            raise ValueError("Shifted crop is empty along one dimension.")

        new_slices.append(slice(start, stop))

    return tuple(new_slices)


def get_sliced_shape(slices, shape):
    """Return the shape obtained by applying ``slices`` to an array of ``shape``.

    Supply one slice per axis.
    """
    out = []
    for sl, dim in zip(slices, shape):
        start, stop, step = sl.indices(dim)
        n = (
            max(0, (stop - start + (step - 1)) // step)
            if step > 0
            else max(0, (start - stop + (-step - 1)) // (-step))
        )
        out.append(n)
    return tuple(out)


def recursive_kmeans(features, *, min_nelem, n_init=200, random_state=0):
    """Recursively cluster features while enforcing a minimum cluster size."""

    n_samp = features.shape[0]
    max_clusters = n_samp // min_nelem
    leaf_clusters = []

    def split(indices):
        if indices.size < 2 * min_nelem:
            leaf_clusters.append(indices.copy())
            return

        upper_k = min(max_clusters, indices.size // min_nelem)
        best_k = 1
        for n_clusters in range(2, upper_k + 1):
            model = KMeans(
                n_clusters=n_clusters,
                n_init=n_init,
                random_state=random_state,
            )
            labels = model.fit_predict(features[indices])
            counts = np.bincount(labels, minlength=n_clusters)
            if np.any(counts < min_nelem):
                break
            best_k = n_clusters

        if best_k < 2:
            leaf_clusters.append(indices.copy())
            return

        model = KMeans(
            n_clusters=best_k,
            n_init=n_init,
            random_state=random_state,
        )
        labels = model.fit_predict(features[indices])
        for cluster_id in range(best_k):
            child_indices = indices[labels == cluster_id]
            if child_indices.size > 0:
                split(child_indices)

    split(np.arange(n_samp))

    labels = -np.ones(n_samp, dtype=int)
    for label, indices in enumerate(leaf_clusters):
        labels[indices] = label
    return labels, leaf_clusters


def dct_highpass(x, *, cutoff, pad_len=None):
    """
    High-pass filter a 1D array via DCT.
    """
    n = x.shape[0]
    if pad_len is None:
        pad_len = n // 2

    x_padded = np.pad(x, pad_width=pad_len, mode="reflect")

    c = dct(x_padded, type=2, norm="ortho")
    c[:cutoff] = 0
    y_padded = idct(c, type=2, norm="ortho")

    return y_padded[pad_len : pad_len + n]


def get_func_ramp(ramp_niter, *, ramp_type="linear"):
    """Return a function mapping iteration index to a learning rate in (0, 1].

    ``ramp_type`` is "linear" or "sin". The rate reaches 1 after
    ``ramp_niter`` ramp iterations; zero disables the ramp.
    """

    if ramp_niter == 0:

        def func_ramp(n: int) -> float:
            return 1.0

        return func_ramp

    ramp_niter = float(ramp_niter)
    ramp_type = ramp_type.lower()
    if ramp_type not in ("linear", "sin"):
        raise ValueError(f"Unknown ramp_type: {ramp_type!r}. Use 'linear' or 'sin'.")

    def func_ramp(n: int) -> float:
        x = (n + 1) / (ramp_niter + 1.0)
        if x >= 1.0:
            return 1.0

        if ramp_type == "linear":
            return float(x)
        else:  # "sin"
            return float(np.sin(0.5 * np.pi * x))

    return func_ramp
