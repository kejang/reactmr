import numpy as np
import cupy as cp
from cupyx.scipy.sparse.linalg import LinearOperator
from numpy.fft import fftshift, fftfreq

from .core import (
    BaseDeviator,
    BaseMotionOp,
    BaseReconstructor,
    react,
    sn_react,
)
from .entropy import EntropyEvaluator
from .partitioners import (
    ClusterCountKMeansPartitioner,
    ContiguousOverlappingBeatPartitioner,
    DummyFeatureExtractor,
    MaxGroupSizeKMeansPartitioner,
    MotionFeatureExtractor,
)
from .solvers import NloptBOBYQASolver
from .cfftn import CupyCenteredFFTN
from .utils import (
    get_roi_slice,
    shift_roi_slice,
    get_sliced_shape,
    get_func_ramp,
)


class Tran2DMotionOp(BaseMotionOp):
    """Apply 2D translations to EPI ky-line data.

    ``sz_im`` and ``spacing`` follow (y, x) array order. Translations use
    physical (x, y, z) coordinates; z is ignored.
    """

    def __init__(self, sz_im, spacing):
        """Set image shape and pixel spacing, both in (y, x) order."""
        ny, nx = sz_im
        res_y, res_x = spacing
        self.freq_x = fftshift(fftfreq(nx)) / res_x
        self.freq_y = fftshift(fftfreq(ny)) / res_y

    def apply_translation(self, data, k, t, *, dtype: str = "complex64"):
        """Return translated ky-line data with output type ``dtype``.

        ``data`` has shape (ncoil, n_intv, nx), and ``k`` contains n_intv
        centered ky row indices. ``t`` is a physical (x, y, z) translation;
        z is ignored.
        """
        tx, ty, _ = t
        phases = np.zeros(data.shape[1:], dtype="float32")
        for i, ind_cdft in enumerate(k):
            phases[i] = 2.0 * np.pi * (self.freq_x * -tx + self.freq_y[ind_cdft] * -ty)
        return (data * np.exp(1j * phases)).astype(dtype)

    def rotate_k(self, k, rotmat):
        """Rotation handling is not implemented for this 2D translation operator."""
        return k

    def rotvec_to_rotmat(self, rotvec):
        """Rotation handling is not implemented for this 2D translation operator."""
        return np.eye(3, dtype="float32")

    def combine_rotvecs(self, inc, base):
        """Rotation handling is not implemented for this 2D translation operator."""
        return base


class Identity(LinearOperator):
    """Identity operator for denoising."""

    def __init__(self, n, *, is_complex, gpu_id, stream):

        self.shape = (n, n)

        if is_complex:
            self.dtype = cp.dtype("complex64")
        else:
            self.dtype = cp.dtype("float32")

        self.gpu_id = gpu_id
        self.stream = stream

    def _matvec(self, x):

        return x

    def _rmatvec(self, x):

        return x


class Tran2DReconstructor(BaseReconstructor):
    """Reconstruct cropped coil-combined images from EPI ky lines.

    ``ncoil`` is the coil count. ``sz_im`` and ``sz_crop`` are full and
    cropped image shapes in (y, x) order; ``roi_slice`` selects the crop.
    ``gpu_id`` and ``stream`` select the CUDA device and stream.
    ``denoise_param`` supplies SplitBregman settings; None disables it.

    Reconstruction methods require ``ws`` for the shared interface,
    although this Cartesian reconstructor does not use the weights.
    """

    def __init__(
        self,
        *,
        ncoil,
        sz_im,
        sz_crop,
        roi_slice,
        gpu_id,
        stream,
        denoise_param=None,
    ):
        """Configure the image geometry, CUDA resources, and optional denoising."""

        self.ncoil = ncoil
        self.sz_im = sz_im
        self.ny = sz_im[0]
        self.sz_crop = sz_crop
        self.roi_slice = roi_slice
        self.gpu_id = gpu_id
        self.stream = stream
        self.cfftn_op = CupyCenteredFFTN(sz_im, gpu_id=gpu_id, stream=stream)
        self.denoise_param = denoise_param
        if denoise_param is not None and denoise_param["niter_outer"] > 0:
            from splitbregman.core import SplitBregman

            self.denoise_op = SplitBregman(
                model=Identity(
                    int(np.prod(self.sz_crop)),
                    is_complex=False,
                    gpu_id=gpu_id,
                    stream=stream,
                ),
                sz_im=self.sz_crop,
                economize_gpu=False,
            )

    def run_sum_of_squares(self, *, denoise: bool = True):

        with cp.cuda.Device(self.gpu_id), self.stream as stream:
            self.d_im = cp.sqrt(
                cp.sum(cp.abs(self.d_ims_bg + self.d_ims_subset) ** 2, axis=0)
            )

        if (
            denoise
            and (self.denoise_param is not None)
            and (self.denoise_param["niter_outer"] > 0)
        ):
            self.d_im = self.denoise_op.run(
                data=self.d_im,
                init=self.d_im,
                mu=self.denoise_param["mu"],
                lams=self.denoise_param["lams"],
                gam=self.denoise_param["gam"],
                niter_inner=self.denoise_param["niter_inner"],
                niter_outer=self.denoise_param["niter_outer"],
                niter_solver=self.denoise_param["niter_solver"],
                wavelet_name=self.denoise_param["wavelet_name"],
                wavelet_full=self.denoise_param["wavelet_full"],
            )

        with cp.cuda.Device(self.gpu_id), self.stream as _:
            self.d_im = self.d_im.reshape(self.sz_crop)

    def prep(self, data, ks, ws):
        """Prepare the background image and clear the subset contribution.

        ``data`` has shape (ncoil, n_beat, n_intv, nx); ``ks`` contains centered
        ky indices of shape (n_beat, n_intv). ``ws`` is required but unused.
        """

        if ws is None:
            raise ValueError("ws must be provided as an array, not None.")
        self.d_ims_bg = self.subset_cifft2(data, ks)
        with cp.cuda.Device(self.gpu_id), self.stream:
            self.d_ims_subset = cp.zeros_like(self.d_ims_bg)

    def recon(self, *, to_host: bool = False, for_scaling: bool = False):
        """Return the cropped sum-of-squares magnitude image.

        ``to_host=True`` returns NumPy data; otherwise it returns CuPy data.
        ``for_scaling=True`` skips denoising.
        """

        if for_scaling:
            denoise = False
        else:
            denoise = True

        self.run_sum_of_squares(denoise=denoise)
        if to_host:
            with cp.cuda.Device(self.gpu_id), self.stream:
                return cp.asnumpy(self.d_im)
        else:
            return self.d_im

    def subset_prep(self, data, ks, ws, *, mode: str = "add"):
        """Add or subtract a ky-line subset from the background image.

        Inputs follow ``prep`` but contain only the subset's beats.
        ``ws`` is required but unused. ``mode`` is "add" or "subtract".
        """
        if ws is None:
            raise ValueError("ws must be provided as an array, not None.")
        self.d_ims_subset = self.subset_cifft2(data, ks)
        with cp.cuda.Device(self.gpu_id), self.stream:
            if mode == "add":
                self.d_ims_bg += self.d_ims_subset
            else:
                self.d_ims_bg -= self.d_ims_subset

    def subset_recon(self, data, ks, ws, *, to_host: bool = False):
        """Return the magnitude image with the candidate ky-line subset included.

        Inputs follow ``prep`` but contain only the subset's beats.
        ``ws`` is required but unused. ``to_host=True`` returns a NumPy array;
        otherwise the result is a CuPy array.
        """

        if ws is None:
            raise ValueError("ws must be provided as an array, not None.")
        self.d_ims_subset = self.subset_cifft2(data, ks)
        self.run_sum_of_squares()
        if to_host:
            with cp.cuda.Device(self.gpu_id), self.stream:
                return cp.asnumpy(self.d_im)
        else:
            return self.d_im

    def subset_cifft2(self, data, ks):
        """Return cropped complex images for each coil from a ky-line subset.

        ``data`` has shape (ncoil, n_group, n_intv, nx); ``ks`` contains centered
        ky indices of shape (n_group, n_intv). Output shape is (ncoil, *sz_crop).
        """

        with cp.cuda.Device(self.gpu_id), self.stream:
            d_ims = cp.empty((self.ncoil, *self.sz_crop), dtype=np.complex64)

        for coil_ind in range(self.ncoil):
            h_data_reordered = np.zeros(self.sz_im, dtype=np.complex64)
            for dft_indices, beat_data in zip(ks, data[coil_ind]):
                h_data_reordered[dft_indices] = beat_data

            with cp.cuda.Device(self.gpu_id), self.stream:
                d_data_reordered = cp.asarray(h_data_reordered, dtype=np.complex64)

            d_im_full = self.cfftn_op.cifftn(d_data_reordered)

            with cp.cuda.Device(self.gpu_id), self.stream:
                d_ims[coil_ind] = d_im_full[self.roi_slice]

        return d_ims


class Tran2DDeviator(BaseDeviator):
    """2D Translation Deviator."""

    def __init__(self, motion_op, *, mode="tran"):

        super().__init__(motion_op, deg_per_cm=0)
        self.set_mode(mode)

    def unscale_dv(self, dv):
        """Convert two optimizer variables to (x, y, 0) translation.

        Returns the translation in physical units and a zero rotation vector,
        each with three components.
        """

        tvec = np.zeros(3, dtype=dv.dtype)
        tvec[:2] = dv
        rotvec = np.zeros(3, dtype=dv.dtype)

        return tvec, rotvec

    def deviate(self, dv):

        dt_vec, _ = self.windowing_and_unscale_dv(dv)
        data = np.zeros_like(self.data_base)
        for i in range(len(self.ks_base)):
            tran_xyz = dt_vec[i] + self.trans[i]
            data[:, i] = self.motion_op.apply_translation(
                self.data_base[:, i],
                self.ks_base[i],  # uncorrected trajectory
                tran_xyz,
            )

        return data, self.ks_base

    def set_mode(self, mode):

        if mode != "tran":
            raise ValueError(f"Unsupported mode: {mode}")
        super().set_mode(mode)
        self.dof_tran = 2
        self.dof_rot = 0


def run_react(
    data_baseline,
    ks,
    *,
    sz_im,
    res_2d,
    niter,
    search_ub_tran,
    search_tol,
    mask=None,
    init_trans=None,
    group_param=None,
    group_input=None,
    init_grid_schedule=None,
    roi_tran=None,
    gpu_id=0,
    denoise_param=None,
    ramp_niter=0,
    ramp_type="linear",
    bg_recompute_interval=0,
    rand_seed=42,
    search_method="BOBYQA",
    verbose=False,
    debug_prefix=None,
    react_type="default",
    sn_param=None,
):
    """Estimate 2D translations from EPI ky-line data.

    Parameters
    ----------
    data_baseline : ndarray
        Uncorrected k-space data, shape (ncoil, n_beat, n_intv, nx).
    ks : ndarray
        Centered ky row indices, shape (n_beat, n_intv).
    sz_im : tuple of int
        Image shape in array order: (ny, nx).
    res_2d : array_like
        Pixel spacing in (x, y) order.
    niter : int
        Number of iterations; outer iterations for self-navigating REACT.
    search_ub_tran : array_like
        Translation search radii in physical units, shape (niter, 2).
    search_tol : float
        Absolute tolerance for the local optimizer.
    mask : ndarray or None, optional
        Objective mask and crop region. None uses the full image.
    init_trans : array_like or None, optional
        Initial translations in physical units, shape (n_beat, 3). None
        starts from zero. Only x and y are used.
    group_param : dict or None, optional
        Beat grouping settings. None or {} disables grouping. Set "method"
        to "contiguous", "kmeans_cluster_count", or
        "kmeans_max_cluster_size". KMeans requires "feature_source":
        "motion", "delta", "motion_and_delta", or "extern". Delta features
        require self-navigating REACT. Other settings belong to the selected
        partitioner.
    group_input : dict or None, optional
        External grouping requires {"features": feature_array}. iNAV
        grouping is unsupported in 2D.
    init_grid_schedule : sequence of dict or None, optional
        One entry per iteration with "dirs", "ranges", and "spacings"; see
        resolve_init_grid_param. None disables grid search for the run or
        for an individual entry.
    roi_tran : array_like or None, optional
        ROI-center shift in physical units, shape (3,). Returned
        translations are corrected for this shift.
    gpu_id : int, optional
        CUDA device ID.
    denoise_param : dict or None, optional
        SplitBregman denoising settings. None disables denoising.
    ramp_niter : int, optional
        Learning-rate ramp duration. Zero disables the ramp.
    ramp_type : {"linear", "sin"}, optional
        Learning-rate ramp shape.
    bg_recompute_interval : int, optional
        Rebuild the background every this many group updates. Zero uses
        incremental updates only.
    rand_seed : int, optional
        Seed for grouping and update order.
    search_method : {"BOBYQA"}, optional
        Local optimizer.
    verbose : bool, optional
        Print progress.
    debug_prefix : str or None, optional
        Prefix for motion and diagnostic files. None disables saving.
    react_type : {"default", "self_navigating"}, optional
        REACT algorithm variant.
    sn_param : dict or None, optional
        Required for self-navigating REACT. Must include "bootstrap_mode"
        and "delta_mode"; see sn_react.

    Returns
    -------
    trans : ndarray, shape (n_beat, 3)
        Estimated translations in physical units.
    rotvecs : ndarray, shape (n_beat, 3)
        Zero rotation vectors.
    iter_costs : ndarray, shape (niter + 1,)
        Objective values before and after each iteration.
    debug_returns : dict
        Motion, grouping, solver, and timing diagnostics.
    """

    if react_type not in ("default", "self_navigating"):
        raise ValueError(
            "react_type must be either 'default' or 'self_navigating', "
            f"got {react_type!r}."
        )
    if react_type == "self_navigating" and sn_param is None:
        raise ValueError("sn_param is required when react_type='self_navigating'.")

    # --- prepare beat partitioning --- #

    group_param = dict(group_param or {})
    group_input = dict(group_input or {})
    group_method = str(group_param.get("method", "none"))
    feature_source = group_param.get("feature_source")

    if group_method not in (
        "none",
        "kmeans_cluster_count",
        "kmeans_max_cluster_size",
        "contiguous",
    ):
        raise ValueError(
            "group_param['method'] must be 'none', 'kmeans_cluster_count', "
            "'kmeans_max_cluster_size', or 'contiguous', "
            f"got {group_method!r}."
        )
    if group_method not in ("none", "contiguous") and feature_source not in (
        "inav",
        "motion",
        "delta",
        "motion_and_delta",
        "extern",
    ):
        raise ValueError(
            "group_param['feature_source'] must be 'inav', 'motion', 'delta', "
            "'motion_and_delta', or 'extern' when grouping is enabled, "
            f"got {feature_source!r}."
        )
    if react_type == "default" and feature_source in (
        "delta",
        "motion_and_delta",
    ):
        raise ValueError(
            f"group_param['feature_source']={feature_source!r} is available "
            "only when react_type='self_navigating'."
        )
    if feature_source in ("delta", "motion_and_delta") and group_method not in (
        "kmeans_cluster_count",
        "kmeans_max_cluster_size",
    ):
        raise ValueError(
            f"group_param['feature_source']={feature_source!r} requires a "
            "KMeans grouping method."
        )

    if group_method not in ("none", "contiguous") and feature_source == "extern":
        if group_input.get("features") is None:
            raise ValueError("External grouping requires group_input['features'].")

    if group_method not in ("none", "contiguous") and feature_source == "inav":
        raise ValueError(
            "iNAV grouping is unsupported for 2D motion; use motion or external "
            "features instead. The iNAV extractor requires 3D translations."
        )

    if group_method == "none":
        partitioner = None
    elif group_method == "contiguous":
        partitioner = ContiguousOverlappingBeatPartitioner(param=group_param)
    else:
        if feature_source in ("motion", "delta", "motion_and_delta"):
            feature_extractor = MotionFeatureExtractor(param=group_param)
        else:
            feature_extractor = DummyFeatureExtractor(
                features=group_input["features"],
            )

        if group_method == "kmeans_cluster_count":
            partitioner = ClusterCountKMeansPartitioner(
                feature_extractor,
                param=group_param,
            )
        else:
            partitioner = MaxGroupSizeKMeansPartitioner(
                feature_extractor,
                param=group_param,
            )

    # --- sizes --- #
    # data: (ncoil, n_beat, n_intv, nx)
    # ks: (n_beat, n_intv)
    # n_beat: number of acquisitions
    # n_intv: EPI factor (number of PE lines per acquisition)

    ncoil, n_beat = data_baseline.shape[:2]
    res_2d = np.asarray(res_2d, dtype="float32")
    spacing = np.array(res_2d[::-1], dtype="float32")  # spacing: (dy, dx)
    ndim = 2  # 2D only

    # --- prepare deviator and motion arithmetic --- #

    deviator = Tran2DDeviator(
        motion_op=Tran2DMotionOp(sz_im, spacing),
    )

    # --- prepare mask --- #

    if mask is not None:
        crop_slice_org = get_roi_slice(mask, margin=1)
        mask_crop = mask[crop_slice_org]
    else:
        crop_slice_org = tuple(slice(0, int(s)) for s in sz_im)
        mask_crop = None

    # --- apply shift to center the roi --- #

    if roi_tran is None:
        roi_tran_px = np.zeros(ndim, dtype="float32")
        roi_tran = np.zeros(3, dtype="float32")
    else:
        roi_tran_px = np.array(roi_tran)[:2] / res_2d

    roi_slice = shift_roi_slice(
        crop_slice_org, np.round(roi_tran_px[::-1]).astype("int32"), sz_im
    )
    sz_crop = get_sliced_shape(roi_slice, sz_im)

    data = np.zeros_like(data_baseline, dtype="complex64")
    for i in range(data_baseline.shape[1]):
        data[:, i] = deviator.motion_op.apply_translation(
            data_baseline[:, i],
            ks[i],
            roi_tran,
        )

    # --- prepare initial translation --- #

    if init_trans is not None:
        init_trans = np.asarray(init_trans).astype("float32")
    else:
        init_trans = np.zeros((n_beat, 3), dtype="float32")

    init_rotvecs = np.zeros((n_beat, 3), dtype="float32")

    # --- prepare reconstructor, entropy evaluator, and solver --- #

    stream = cp.cuda.stream.Stream(non_blocking=True)

    entropy_evaluator = EntropyEvaluator(
        mask_crop,
        sz_crop=sz_crop,
        is_complex=False,  # is_complex (SoS is real-valued)
        gpu_id=gpu_id,
        stream=stream,
    )

    reconstructor = Tran2DReconstructor(
        ncoil=ncoil,
        sz_im=sz_im,
        sz_crop=sz_crop,
        roi_slice=roi_slice,
        gpu_id=gpu_id,
        stream=stream,
        denoise_param=denoise_param,
    )

    if search_method == "BOBYQA":
        common_solver_param = {
            "ftol_abs": 1e-6,
            "maxeval": 200,
            "fallback_on_roundoff": True,
        }
        solver = NloptBOBYQASolver(
            xtol_abs=search_tol,
            **common_solver_param,
        )
    else:
        raise ValueError(f"Unknown search method: {search_method}")

    # --- scale data (for denoising parameter) --- #

    ws = np.ones_like(ks, dtype="float32")  # dummy weights for the shared interface
    reconstructor.prep(data, ks, ws)
    h_im = reconstructor.recon(to_host=True, for_scaling=True)
    h_mask = getattr(entropy_evaluator, "h_mask", None)
    if h_mask is not None:
        binary = np.extract(h_mask > 0, h_im)
    else:
        binary = h_im.flatten()
    norm_factor = np.percentile(binary, 99.0)
    data /= norm_factor

    # --- prepare learning rate function --- #

    func_learning_rate = get_func_ramp(ramp_niter, ramp_type=ramp_type)

    # --- unused parameters --- #

    search_ub_rot = np.zeros(
        (niter, 1), dtype="float32"
    )  # no rotation in 2D translation

    # --- run react --- #

    if react_type == "default":
        trans_offcntr, rotvecs, iter_costs, debug_returns = react(
            data,
            ks,
            ws,
            trans=init_trans,
            rotvecs=init_rotvecs,
            niter=niter,
            search_ub_tran=search_ub_tran,
            search_ub_rot=search_ub_rot,
            init_grid_schedule=init_grid_schedule,
            entropy_evaluator=entropy_evaluator,
            deviator=deviator,
            reconstructor=reconstructor,
            solver=solver,
            partitioner=partitioner,
            func_learning_rate=func_learning_rate,
            bg_recompute_interval=bg_recompute_interval,
            rand_seed=rand_seed,
            verbose=verbose,
            debug_prefix=debug_prefix,
        )
    else:
        trans_offcntr, rotvecs, iter_costs, debug_returns = sn_react(
            data,
            ks,
            ws,
            trans=init_trans,
            rotvecs=init_rotvecs,
            niter=niter,
            search_ub_tran=search_ub_tran,
            search_ub_rot=search_ub_rot,
            init_grid_schedule=init_grid_schedule,
            entropy_evaluator=entropy_evaluator,
            deviator=deviator,
            reconstructor=reconstructor,
            solver=solver,
            partitioner=partitioner,
            func_learning_rate=func_learning_rate,
            sn_param=sn_param,
            bg_recompute_interval=bg_recompute_interval,
            rand_seed=rand_seed,
            verbose=verbose,
            debug_prefix=debug_prefix,
        )

    # --- compensate for roi center shift --- #

    trans = trans_offcntr - roi_tran

    return trans, rotvecs, iter_costs, debug_returns
