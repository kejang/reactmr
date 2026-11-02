import numpy as np
from scipy.spatial.transform import Rotation
import cupy as cp

from .core import (
    BaseDeviator,
    BaseMotionOp,
    react,
    sn_react,
)
from .entropy import EntropyEvaluator
from .partitioners import (
    ClusterCountKMeansPartitioner,
    ContiguousOverlappingBeatPartitioner,
    DummyFeatureExtractor,
    InavFeatureExtractor,
    MaxGroupSizeKMeansPartitioner,
    MotionFeatureExtractor,
    SubatomicKMeansBeatPartitioner,
)
from .solvers import NloptBOBYQASolver
from .reconstructors import SoSReconstructor
from .utils import (
    get_roi_slice,
    shift_roi_slice,
    get_sliced_shape,
    get_func_ramp,
)


class Rigid3DMotionOp(BaseMotionOp):
    """3D motion using (x, y, z) translations and rotation vectors in radians."""

    def rotvec_to_rotmat(self, rotvec):
        """Return a 3x3 rotation matrix from a 3-element vector in radians."""
        return Rotation.from_rotvec(rotvec).as_matrix()

    def combine_rotvecs(self, inc, base):
        """Compose rotations as ``R_inc @ R_base`` and return the vector in radians."""
        r_base = Rotation.from_rotvec(base)
        r_inc = Rotation.from_rotvec(inc)
        r_new = r_inc * r_base
        return r_new.as_rotvec()

    def apply_translation(self, data, k, t, *, dtype="complex64"):
        """Return k-space data translated by ``t`` with output type ``dtype``.

        ``k`` has shape (..., 3) in cycles per physical unit; ``t`` has
        shape (3,) in physical units. Axes follow physical coordinate order.
        """
        k = np.asarray(k)
        t = np.asarray(t)
        phase = 2.0 * np.pi * np.inner(k, -t)
        return (data * np.exp(1j * phase)).astype(dtype)

    def rotate_k(self, k: np.ndarray, rotmat: np.ndarray) -> np.ndarray:
        """Return rotated k-space coordinates.

        ``k`` has shape (..., 3); ``rotmat`` has shape (3, 3).
        """

        flat = np.asarray(k).reshape(-1, rotmat.shape[0])
        flat_rot = (rotmat @ flat.T).T
        k_rot = flat_rot.reshape(k.shape)

        return k_rot


class Rigid3DDeviator(BaseDeviator):
    """Convert optimizer variables to 3D motion increments.

    ``mode="tran"`` optimizes translations only; "rigid" adds rotations.
    ``deg_per_cm`` sets rotation scaling. Translations have 3 components
    and rotations have 3, in radians.
    """

    def __init__(self, motion_op, *, deg_per_cm, mode="rigid"):

        super().__init__(motion_op, deg_per_cm=deg_per_cm)
        self.set_mode(mode)

    def set_mode(self, mode):

        super().set_mode(mode)
        self.dof_tran = 3
        self.dof_rot = 3 if mode == "rigid" else 0

    def unscale_dv(self, dv):
        """Convert optimizer variables ``dv`` to physical motion increments.

        Return 3 translation components in physical units and 3 rotation
        components in radians.
        """

        if self.mode == "tran":
            tvec = dv.copy()
            rotvec = np.zeros(3, dtype=dv.dtype)
        else:  # default is rigid
            tvec = dv[: self.dof_tran].copy()
            rotvec = self.rot_scale * dv[self.dof_tran :].copy()

        return tvec, rotvec


def run_react(
    data_baseline,
    ks_baseline,
    ws,
    *,
    sz_im,
    res,
    niter,
    search_ub_tran,
    search_ub_rot,
    search_tol,
    mask=None,
    init_trans=None,
    init_rotvecs=None,
    group_param=None,
    group_input=None,
    init_grid_schedule=None,
    roi_tran=None,
    osf=1.15,
    kernel_width=6.0,
    sens=None,
    gpu_id=0,
    mode="rigid",
    denoise_param=None,
    deg_per_cm=10.0,
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
    """Estimate 3D rigid motion with REACT.

    Parameters
    ----------
    data_baseline : ndarray
        Uncorrected k-space data, shape (ncoil, n_beat, ...).
    ks_baseline : ndarray
        Uncorrected trajectory, shape (n_beat, ..., 3).
    ws : ndarray
        Required density compensation weights, shape (n_beat, ...).
    sz_im : tuple of int
        Image shape in array order: (nz, ny, nx).
    res : array_like
        Voxel spacing in (x, y, z) order.
    niter : int
        Number of iterations; outer iterations for self-navigating REACT.
    search_ub_tran : array_like
        Translation search radii in physical units, shape (niter, 3).
    search_ub_rot : array_like
        Rotation search radii in radians, shape (niter, 3).
    search_tol : float
        Absolute tolerance for the local optimizer.
    mask : ndarray or None, optional
        Objective mask and crop region. None uses the full image.
    init_trans : array_like or None, optional
        Initial translations in physical units, shape (n_beat, 3). None
        starts from zero.
    init_rotvecs : array_like or None, optional
        Initial rotations in radians, shape (n_beat, 3). None starts from
        zero.
    group_param : dict or None, optional
        Beat grouping settings. None or {} disables grouping. Set "method"
        to "contiguous", "kmeans_cluster_count", or
        "kmeans_max_cluster_size". "subatomic_kmeans" is also supported; see
        SubatomicKMeansBeatPartitioner. KMeans requires "feature_source":
        "motion", "delta", "motion_and_delta", "extern", or "inav". Delta
        features require self-navigating REACT. Other settings belong to the
        selected partitioner.
    group_input : dict or None, optional
        External grouping requires "features". iNAV grouping requires
        "inavs", "inav_masks", "inav_spacings", and "inav_planes".
    init_grid_schedule : sequence of dict or None, optional
        One entry per iteration with "dirs", "ranges", and "spacings"; see
        resolve_init_grid_param. None disables grid search for the run or
        for an individual entry.
    roi_tran : array_like or None, optional
        ROI-center shift in physical units, shape (3,). Returned
        translations are corrected for this shift.
    osf, kernel_width : float, optional
        Gridding oversampling factor and kernel width.
    sens : ndarray or None, optional
        Coil sensitivity maps.
    gpu_id : int, optional
        CUDA device ID.
    mode : {"rigid", "tran"} or sequence of str, optional
        One motion mode for the whole run, or one per iteration.
    denoise_param : dict or None, optional
        SplitBregman denoising settings. None disables denoising.
    deg_per_cm : float, optional
        Rotation-to-translation scaling for optimization, in degrees per cm.
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
        Estimated rotations in radians.
    iter_costs : ndarray, shape (niter + 1,)
        Objective values before and after each iteration.
    debug_returns : dict
        Motion, grouping, solver, and timing diagnostics.

    Notes
    -----
    Subatomic grouping supports default REACT only. Input data must have
    shape (ncoil, n_heartbeat, n_trs_per_beat, ...). Initial motion is per
    heartbeat; returned motion has one row per TR. niter must equal the sum
    of the four stage counts.
    """

    if ws is None:
        raise ValueError("ws must be provided as an array, not None.")

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
    has_subatomic_stages = any(
        name in group_param
        for name in ("atomic_kmeans", "atomic", "subatomic_kmeans", "subatomic")
    )
    default_method = "subatomic_kmeans" if has_subatomic_stages else "none"
    group_method = str(group_param.get("method", default_method))
    if group_method not in (
        "none",
        "kmeans_cluster_count",
        "kmeans_max_cluster_size",
        "contiguous",
        "subatomic_kmeans",
    ):
        raise ValueError(f"Unknown grouping method: {group_method!r}.")
    is_subatomic = group_method == "subatomic_kmeans"
    if is_subatomic and react_type != "default":
        raise ValueError("Subatomic grouping is supported only by default REACT.")

    def make_feature_extractor(feature_param):
        """Construct an extractor from the selected feature settings."""
        source = feature_param.get("feature_source")
        if source not in ("inav", "motion", "delta", "motion_and_delta", "extern"):
            raise ValueError(
                "feature_source must be 'inav', 'motion', 'delta', "
                f"'motion_and_delta', or 'extern', got {source!r}."
            )
        if source in ("delta", "motion_and_delta") and react_type == "default":
            raise ValueError(
                f"feature_source={source!r} requires self-navigating REACT."
            )
        if source in ("motion", "delta", "motion_and_delta"):
            return MotionFeatureExtractor(param=feature_param)
        if source == "extern":
            if group_input.get("features") is None:
                raise ValueError("External grouping requires group_input['features'].")
            return DummyFeatureExtractor(features=group_input["features"])

        required_inputs = ("inavs", "inav_masks", "inav_spacings", "inav_planes")
        missing_inputs = [
            name for name in required_inputs if group_input.get(name) is None
        ]
        if missing_inputs:
            raise ValueError(
                "iNAV grouping requires the following group_input entries: "
                + ", ".join(missing_inputs)
            )
        return InavFeatureExtractor(
            **{name: group_input[name] for name in required_inputs},
            param=feature_param,
        )

    feature_source = group_param.get("feature_source")
    if group_method == "none":
        partitioner = None
    elif group_method == "contiguous":
        if feature_source in ("delta", "motion_and_delta"):
            raise ValueError("Delta feature sources require a KMeans grouping method.")
        partitioner = ContiguousOverlappingBeatPartitioner(param=group_param)
    elif group_method == "subatomic_kmeans":
        atomic_kmeans_param = dict(group_param.get("atomic_kmeans") or {})
        subatomic_kmeans_param = dict(group_param.get("subatomic_kmeans") or {})
        atomic_kmeans_param.setdefault("niter", 0)
        atomic_kmeans_param.setdefault("feature_source", "motion")
        subatomic_kmeans_param.setdefault("niter", 0)
        feature_source = atomic_kmeans_param["feature_source"]
        subatomic_kmeans_param["feature_source"] = feature_source
        group_param["atomic_kmeans"] = atomic_kmeans_param
        group_param["subatomic_kmeans"] = subatomic_kmeans_param
        feature_extractor = (
            make_feature_extractor(atomic_kmeans_param)
            if int(atomic_kmeans_param["niter"]) > 0
            or int(subatomic_kmeans_param["niter"]) > 0
            else None
        )
        partitioner = SubatomicKMeansBeatPartitioner(
            feature_extractor=feature_extractor, param=group_param
        )
    else:
        feature_extractor = make_feature_extractor(group_param)
        if group_method == "kmeans_cluster_count":
            partitioner = ClusterCountKMeansPartitioner(
                feature_extractor, param=group_param
            )
        else:
            partitioner = MaxGroupSizeKMeansPartitioner(
                feature_extractor, param=group_param
            )

    # --- flatten heartbeat/TR axes for subatomic estimation --- #

    if is_subatomic:
        n_trs_per_beat = partitioner.n_trs_per_beat
        n_heartbeat = data_baseline.shape[1]
        if data_baseline.ndim < 4 or data_baseline.shape[2] != n_trs_per_beat:
            raise ValueError(
                "Subatomic data must have shape (ncoil, n_heartbeat, "
                "n_trs_per_beat, ...)."
            )
        if ks_baseline.shape != (*data_baseline.shape[1:], 3):
            raise ValueError("Subatomic trajectories must align with the data axes.")
        if ws.shape != data_baseline.shape[1:]:
            raise ValueError("Subatomic weights must align with the data axes.")
        if niter != partitioner.inner_niter:
            raise ValueError(
                "niter must equal the sum of the configured stage iteration counts."
            )

        n_tr = n_heartbeat * n_trs_per_beat
        data_baseline = data_baseline.reshape(
            data_baseline.shape[0], n_tr, *data_baseline.shape[3:]
        )
        ks_baseline = ks_baseline.reshape(n_tr, *ks_baseline.shape[2:])
        ws = ws.reshape(n_tr, *ws.shape[2:])
        if init_trans is not None:
            init_trans = np.asarray(init_trans)
            if init_trans.shape != (n_heartbeat, 3):
                raise ValueError(
                    "Subatomic init_trans must have shape (n_heartbeat, 3)."
                )
            init_trans = np.repeat(init_trans, n_trs_per_beat, axis=0)
        if init_rotvecs is not None:
            init_rotvecs = np.asarray(init_rotvecs)
            if init_rotvecs.shape != (n_heartbeat, 3):
                raise ValueError(
                    "Subatomic init_rotvecs must have shape (n_heartbeat, 3)."
                )
            init_rotvecs = np.repeat(init_rotvecs, n_trs_per_beat, axis=0)

    # --- sizes --- #
    # data: (ncoil, n_beat, n_intv, npt) (or ncoil, n_beat, ..., npt)
    # ks: (n_beat, n_intv, npt, 3) (or n_beat, ..., npt, 3)
    # ws: (n_beat, n_intv, npt) (or n_beat, ..., npt)
    ncoil = data_baseline.shape[0]
    n_beat = data_baseline.shape[1]
    ndim = 3  # 3D only
    res = np.array(res, dtype="float32")

    # --- prepare deviator and motion arithmetic --- #

    deviator_mode = mode if isinstance(mode, str) else "rigid"
    deviator = Rigid3DDeviator(
        motion_op=Rigid3DMotionOp(),
        deg_per_cm=deg_per_cm,
        mode=deviator_mode,
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
        roi_tran = np.zeros(ndim, dtype="float32")
    else:
        roi_tran_px = np.array(roi_tran) / res

    roi_slices = shift_roi_slice(
        crop_slice_org, np.round(roi_tran_px[::-1]).astype("int32"), sz_im
    )
    sz_crop = get_sliced_shape(roi_slices, sz_im)

    data = np.zeros_like(data_baseline, dtype="complex64")
    for i in range(data_baseline.shape[1]):
        data[:, i] = deviator.motion_op.apply_translation(
            data_baseline[:, i],
            ks_baseline[i],
            roi_tran,
        )

    # --- prepare initial translation --- #

    if init_trans is not None:
        init_trans = np.asarray(init_trans).astype("float32")
    else:
        init_trans = np.zeros((n_beat, 3), dtype="float32")

    if init_rotvecs is not None:
        init_rotvecs = np.asarray(init_rotvecs).astype("float32")
    else:
        init_rotvecs = np.zeros((n_beat, 3), dtype="float32")

    # --- prepare reconstructor, entropy evaluator, and solver --- #

    stream = cp.cuda.stream.Stream(non_blocking=True)

    entropy_evaluator = EntropyEvaluator(
        mask_crop=mask_crop,
        sz_crop=sz_crop,
        is_complex=False,  # (SoS is real-valued)
        gpu_id=gpu_id,
        stream=stream,
    )

    reconstructor = SoSReconstructor(
        sz_crop=sz_crop,
        roi_slices=roi_slices,
        ncoil=ncoil,
        sz_im=sz_im,
        res=res,
        ks=ks_baseline,
        ws=ws,
        sens=sens,
        osf=osf,
        kernel_width=kernel_width,
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

    reconstructor.prep(data, ks_baseline, ws)
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

    # --- run react --- #

    if react_type == "default":
        trans_offcntr, rotvecs, iter_costs, debug_returns = react(
            data,
            ks_baseline,
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
            modes=mode,
            func_learning_rate=func_learning_rate,
            bg_recompute_interval=bg_recompute_interval,
            rand_seed=rand_seed,
            verbose=verbose,
            debug_prefix=debug_prefix,
        )
    else:
        trans_offcntr, rotvecs, iter_costs, debug_returns = sn_react(
            data,
            ks_baseline,
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
            modes=mode,
            func_learning_rate=func_learning_rate,
            sn_param=sn_param,
            bg_recompute_interval=bg_recompute_interval,
            rand_seed=rand_seed,
            verbose=verbose,
            debug_prefix=debug_prefix,
        )
    debug_returns["partitioning_method"] = group_method
    debug_returns["partitioning_feature_source"] = feature_source
    if is_subatomic:
        debug_returns["n_heartbeats"] = n_heartbeat
        debug_returns["n_trs_per_beat"] = n_trs_per_beat
        debug_returns["group_param"] = group_param

    # --- compensate for roi center shift --- #

    cntr_trans = np.zeros_like(trans_offcntr)
    for i, rotvec in enumerate(rotvecs):
        rotmat = deviator.motion_op.rotvec_to_rotmat(rotvec)
        cntr_trans[i] = (
            rotmat @ roi_tran
        )  # translation in the lab frame due to rotation

    trans = trans_offcntr + cntr_trans - roi_tran

    return trans, rotvecs, iter_costs, debug_returns
