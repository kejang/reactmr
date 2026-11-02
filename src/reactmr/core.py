from abc import ABC, abstractmethod
import pickle
import time

import numpy as np

from .grid_search import resolve_init_grid_param
from .partitioners import BeatPartition, PartitionStep


class BaseReconstructor(ABC):
    """Interface for full-data reconstruction and subset updates."""

    @abstractmethod
    def prep(self, data, ks, ws):
        """Initialize reconstruction from motion-corrected data.

        ``data`` has shape (ncoil, n_beat, ...). ``ks`` gives the sampling
        coordinates and ``ws`` must contain matching weights.
        """
        ...

    @abstractmethod
    def recon(self, *, to_host: bool = False):
        """Return the image from the prepared reconstruction state.

        ``to_host=True`` requests a NumPy array; otherwise the result may
        remain on the reconstruction device.
        """
        ...

    @abstractmethod
    def subset_prep(self, data, ks, ws, *, mode: str = "add"):
        """Add or subtract a subset's contribution to the background.

        ``data``, ``ks``, and required ``ws`` describe the same subset.
        Set ``mode`` to "add" or "subtract".
        """
        ...

    @abstractmethod
    def subset_recon(self, data, ks, ws):
        """Return an image with the candidate subset included.

        ``data``, ``ks``, and required ``ws`` describe the candidate subset.
        """
        ...


class BaseEntropyEvaluator(ABC):
    """Abstract interface for an image-quality metric used by REACT."""

    @abstractmethod
    def eval(self, im) -> float:
        """Return the image-quality cost for ``im`` as a float."""
        ...


class BaseMotionOp(ABC):
    """Interface for rotation composition and k-space motion transforms."""

    @abstractmethod
    def rotvec_to_rotmat(self, rotvec):
        """Return the rotation matrix for a rotation vector in radians."""
        ...

    @abstractmethod
    def combine_rotvecs(self, inc, base):
        """Return the composition of incremental and base rotations, in radians.

        The order is ``R_inc @ R_base``.
        """
        ...

    @abstractmethod
    def rotate_k(self, k, rotmat):
        """Rotate k-space coordinates ``k`` using ``rotmat``.

        Shapes are (..., ndim) and (ndim, ndim); the result has ``k.shape``.
        """
        ...

    @abstractmethod
    def apply_translation(self, data, k, t, *, dtype: str = "complex64"):
        """Return k-space data after applying translation ``t``.

        ``k`` describes the samples; ``t`` uses the motion model's physical
        units and axis order. ``dtype`` sets the output data type.
        """
        ...


class BaseDeviator(ABC):
    """Map optimizer variables to physical motion and k-space updates.

    ``motion_op`` supplies the motion transforms. ``deg_per_cm`` sets
    rotation-to-translation scaling in degrees per cm.
    """

    def __init__(
        self,
        motion_op: BaseMotionOp,
        *,
        deg_per_cm: float,
    ):

        self.motion_op = motion_op  # motion arithmetic

        # solver sees (deg_per_cm) * [degree]
        # radian -> solver: rotvec / rot_scale
        # solver -> radian: solution * rot_scale
        rot_scale = deg_per_cm * (np.pi / 180.0)
        self.rot_scale = rot_scale

    @abstractmethod
    def unscale_dv(self, dv):
        """Convert optimizer variables to translation and rotation increments.

        Returns translations in physical units and rotations in radians;
        vector lengths depend on the motion model.
        """
        ...

    def windowing_and_unscale_dv(self, dv):
        """Return windowed physical increments with one row per subset beat."""
        dt_vec, dr_vec = self.unscale_dv(dv)
        window = np.asarray(self.group_window, dtype=dt_vec.dtype)[:, None]
        return window * dt_vec[None, :], window * dr_vec[None, :]

    def deviate(self, dv):
        """Return candidate data and sampling coordinates for increment ``dv``.

        Call ``set_state`` first. Rotations update the current trajectory;
        translation phases use the baseline trajectory.
        """
        dt_vec, dr_vec = self.windowing_and_unscale_dv(dv)

        ks = np.zeros_like(self.ks_base)
        data = np.zeros_like(self.data_base)
        for i in range(len(self.ks_base)):
            # further rotate ks (corrected trajectory)
            rotmat = self.motion_op.rotvec_to_rotmat(dr_vec[i])
            ks[i] = self.motion_op.rotate_k(self.ks[i], rotmat)

            # additional phase
            tran_xyz = dt_vec[i] + self.trans[i]
            data[:, i] = self.motion_op.apply_translation(
                self.data_base[:, i],
                self.ks_base[i],  # uncorrected trajectory
                tran_xyz,
            )

        return data, ks

    def set_state(self, data_base, ks_base, ks, trans, group_window):
        """Store the subset used by subsequent ``deviate`` calls.

        ``data_base`` and ``ks_base`` are uncorrected inputs. ``ks`` and
        ``trans`` are the current trajectory and translations.
        ``group_window`` contains one update weight per subset beat.
        All inputs are copied.
        """
        self.data_base = data_base.copy()
        self.ks_base = ks_base.copy()
        self.ks = ks.copy()
        self.trans = trans.copy()
        self.group_window = group_window.copy()

    def set_mode(self, mode):
        """Select "tran" or "rigid" motion; raise ValueError for other modes."""
        if mode not in ("tran", "rigid"):
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode


class BaseSolver(ABC):
    """Abstract interface for local optimization of motion increments."""

    def __init__(self, *, name: str = "unknown"):
        self.name = name

    @abstractmethod
    def solve(self, fun, x0, *, lower_bounds, upper_bounds):
        """Minimize ``fun(x)`` within the supplied bounds, starting at ``x0``.

        Bounds must match ``x0``. Return the solution array and a dictionary
        of solver diagnostics.
        """
        ...


def react_step(
    trans,
    rotvecs,
    data,
    ks,
    *,
    partition_step,
    data_baseline,
    ks_baseline,
    ws,
    entropy_evaluator,
    deviator,
    reconstructor,
    solver,
    search_ub_tran,
    search_ub_rot,
    init_grid_param,
    learning_rate,
    bg_recompute_interval=0,
):
    """Update motion for one iteration and return increments and diagnostics.

    Modifies ``trans``, ``rotvecs``, ``data``, and ``ks`` in place.
    ``data_baseline`` and ``ks_baseline`` are the uncorrected inputs;
    ``ws`` must contain the corresponding weights.

    ``partition_step`` supplies groups, windows, and update settings.
    ``init_grid_param`` contains numeric ``dirs``, ``ranges``, and
    ``spacings``, or None to skip grid search. ``learning_rate`` scales
    each update. Other search and reconstruction arguments follow ``react``.

    Returns translation increments, rotation increments (radians), and a
    diagnostic dictionary. Increment arrays match the input motion shapes.
    """

    if ws is None:
        raise ValueError("ws must be provided as an array, not None.")

    partitions = partition_step.partitions
    take_mean_partitions = partition_step.take_mean_partitions
    freeze_reference_group = partition_step.freeze_reference_group
    debug_return = {"partition_steps": []}

    # ---- determine search range ---- #

    if deviator.mode == "rigid":
        upper_bounds_base = np.concatenate(
            [
                search_ub_tran,
                search_ub_rot / deviator.rot_scale,
            ]
        )
    elif deviator.mode == "tran":
        upper_bounds_base = search_ub_tran.copy()
    else:
        raise ValueError(f"Unknown mode: {deviator.mode}")

    lower_bounds_base = -upper_bounds_base.copy()

    reconstructor.prep(data, ks, ws)

    # ---- update partitions ---- #

    if take_mean_partitions and len(partitions) > 1:
        trans_partition_base = trans.copy()
        rotvecs_partition_base = rotvecs.copy()
        data_partition_base = data.copy()
        ks_partition_base = ks.copy()

    partition_delta_trans = []
    partition_delta_rotvecs = []

    for partition_index, partition in enumerate(partitions):
        if take_mean_partitions and partition_index > 0:
            trans[:] = trans_partition_base
            rotvecs[:] = rotvecs_partition_base
            data[:] = data_partition_base
            ks[:] = ks_partition_base
        reconstructor.prep(data, ks, ws)

        group_delta_trans = np.zeros_like(trans)
        group_delta_rotvecs = np.zeros_like(rotvecs)
        partition_debug = {}
        trans_grid_search = np.full_like(trans, np.nan)
        n_group_updates = 0

        for group_index, beat_indices in enumerate(partition.groups):
            if freeze_reference_group and group_index == partition.ref_group_index:
                continue

            # prepare the group model and subtract its current contribution

            group_ws = ws[beat_indices]
            reconstructor.subset_prep(
                data[:, beat_indices],
                ks[beat_indices],
                group_ws,
                mode="subtract",
            )

            # set current state

            deviator.set_state(
                data_baseline[:, beat_indices],
                ks_baseline[beat_indices],
                ks[beat_indices],
                trans[beat_indices],
                partition.group_windows[group_index],
            )

            # initial guess

            if deviator.mode == "rigid":
                x0 = np.zeros(deviator.dof_tran + deviator.dof_rot, dtype="float32")
            else:  # tran or rot
                x0 = np.zeros(deviator.dof_tran, dtype="float32")

            # --- 1D grid search for translation (optional) --- #

            if init_grid_param is not None:
                orig_mode = deviator.mode
                deviator.set_mode("tran")  # temporarily set to translation-only

                grid_dirs = init_grid_param["dirs"]
                grid_ranges = init_grid_param["ranges"]
                grid_spacings = init_grid_param["spacings"]
                tran_init = np.zeros(deviator.dof_tran, dtype="float32")
                for ind_dir, direction in enumerate(grid_dirs):
                    step = grid_spacings[ind_dir]
                    if (step is None) or (step < 1e-9):
                        continue

                    ub = grid_ranges[ind_dir]
                    num = int(np.floor(ub / step))
                    if num < 1:
                        continue

                    ds = step * np.arange(-num, num + 1)
                    scores = []
                    for d in ds:
                        d_test = tran_init + d * direction
                        data_dev, ks_dev = deviator.deviate(d_test)
                        recon = reconstructor.subset_recon(data_dev, ks_dev, group_ws)
                        scores.append(entropy_evaluator.eval(recon))

                    ind_min = int(np.argmin(scores))
                    tran_init += ds[ind_min] * direction

                x0[: deviator.dof_tran] = tran_init

                deviator.set_mode(orig_mode)  # restore original mode

            lower_bounds = x0 + lower_bounds_base
            upper_bounds = x0 + upper_bounds_base

            grid_trans, _ = deviator.windowing_and_unscale_dv(x0)
            trans_grid_search[beat_indices] = grid_trans

            # solve for incremental translation and rotation

            if solver is None:
                sol = x0.copy()
                dbg_solver = {}
            else:

                def fun(x):
                    data_dev, ks_dev = deviator.deviate(x)
                    recon = reconstructor.subset_recon(data_dev, ks_dev, group_ws)
                    return entropy_evaluator.eval(recon)

                sol, dbg_solver = solver.solve(
                    fun, x0, lower_bounds=lower_bounds, upper_bounds=upper_bounds
                )

            for dbg_key in dbg_solver:
                if dbg_key not in partition_debug:
                    partition_debug[dbg_key] = []
                partition_debug[dbg_key].append(dbg_solver[dbg_key])

            # ---- update ---- #

            sol *= learning_rate
            small_tran, small_rotvec = deviator.windowing_and_unscale_dv(sol)
            data[:, beat_indices], ks[beat_indices] = deviator.deviate(sol)

            # accumulated translation and rotation

            trans[beat_indices] += small_tran
            for i, beat_index in enumerate(beat_indices):
                rotvecs[beat_index] = deviator.motion_op.combine_rotvecs(
                    small_rotvec[i],
                    rotvecs[beat_index],
                )

            # update background

            n_group_updates += 1
            recompute_bg = (
                bg_recompute_interval > 0
                and n_group_updates % bg_recompute_interval == 0
            )
            if recompute_bg:
                reconstructor.prep(data, ks, ws)
            else:
                reconstructor.subset_prep(
                    data[:, beat_indices],
                    ks[beat_indices],
                    group_ws,
                    mode="add",
                )

            # Accumulate this partition's increments per beat so the bookkeeping
            # remains correct even if groups overlap.

            group_delta_trans[beat_indices] += small_tran
            for i, ind_beat in enumerate(beat_indices):
                group_delta_rotvecs[ind_beat] = deviator.motion_op.combine_rotvecs(
                    small_rotvec[i],
                    group_delta_rotvecs[ind_beat],
                )

        partition_debug["trans_grid_search"] = trans_grid_search
        partition_delta_trans.append(group_delta_trans)
        partition_delta_rotvecs.append(group_delta_rotvecs)
        debug_return["partition_steps"].append(partition_debug)

    if not take_mean_partitions:
        # Sequential updates are already applied; return their total increments.
        new_delta_trans = np.sum(partition_delta_trans, axis=0)
        new_delta_rotvecs = np.zeros_like(rotvecs)
        for delta_rotvecs in partition_delta_rotvecs:
            for ind_beat in range(trans.shape[0]):
                new_delta_rotvecs[ind_beat] = deviator.motion_op.combine_rotvecs(
                    delta_rotvecs[ind_beat],
                    new_delta_rotvecs[ind_beat],
                )
        return new_delta_trans, new_delta_rotvecs, debug_return

    mean_delta_trans = np.mean(partition_delta_trans, axis=0)
    mean_delta_rotvecs = np.mean(partition_delta_rotvecs, axis=0)

    if len(partitions) > 1:
        trans[:] = trans_partition_base
        rotvecs[:] = rotvecs_partition_base
        data[:] = data_partition_base
        ks[:] = ks_partition_base

        for ind_beat in range(trans.shape[0]):
            trans[ind_beat] += mean_delta_trans[ind_beat]
            rotvecs[ind_beat] = deviator.motion_op.combine_rotvecs(
                mean_delta_rotvecs[ind_beat],
                rotvecs[ind_beat],
            )
            ks[ind_beat] = deviator.motion_op.rotate_k(
                ks[ind_beat],
                deviator.motion_op.rotvec_to_rotmat(mean_delta_rotvecs[ind_beat]),
            )
            data[:, ind_beat] = deviator.motion_op.apply_translation(
                data_baseline[:, ind_beat],
                ks_baseline[ind_beat],
                trans[ind_beat],
            )

        reconstructor.prep(data, ks, ws)

    new_delta_trans = mean_delta_trans
    new_delta_rotvecs = np.zeros_like(rotvecs)
    for ind_beat in range(trans.shape[0]):
        new_delta_rotvecs[ind_beat] = deviator.motion_op.combine_rotvecs(
            mean_delta_rotvecs[ind_beat],
            new_delta_rotvecs[ind_beat],
        )

    return new_delta_trans, new_delta_rotvecs, debug_return


def _prepare_debug_returns(
    niter,
    trans,
    rotvecs,
    modes,
    partitioner,
    extra_fields=None,
):
    """Allocate shared motion, timing, and partition diagnostics."""
    n_beat = trans.shape[0]
    debug_returns = {
        "trans": np.zeros((niter, n_beat, trans.shape[-1]), dtype="float32"),
        "rotvecs": np.zeros((niter, n_beat, rotvecs.shape[-1]), dtype="float32"),
        "elapsed_times": np.zeros(niter, dtype="float32"),
        "modes": modes.copy(),
        "n_groups": [None] * niter,
        "ref_group_indices": [None] * niter,
        "partitioning_mode": "none" if partitioner is None else "partitioner",
    }
    if extra_fields is not None:
        debug_returns.update(extra_fields)

    return debug_returns


def _record_debug_iteration(
    debug_returns,
    ind_iter,
    iter_costs,
    trans,
    rotvecs,
    elapsed_time,
    partitions,
    step_debug,
):
    """Record shared motion, partition, and step diagnostics for one iteration."""
    debug_returns["iter_costs"] = iter_costs
    debug_returns["trans"][ind_iter] = trans
    debug_returns["rotvecs"][ind_iter] = rotvecs
    debug_returns["elapsed_times"][ind_iter] = elapsed_time
    debug_returns["n_groups"][ind_iter] = [
        len(partition.groups) for partition in partitions
    ]
    debug_returns["ref_group_indices"][ind_iter] = [
        -1 if partition.ref_group_index is None else partition.ref_group_index
        for partition in partitions
    ]

    niter = len(debug_returns["elapsed_times"])
    for key, value in step_debug.items():
        if key not in debug_returns:
            debug_returns[key] = [[] for _ in range(niter)]
        debug_returns[key][ind_iter] = value


def _save_debug_outputs(
    debug_prefix,
    ind_iter,
    iter_costs,
    trans,
    rotvecs,
    debug_returns,
):
    """Write the per-iteration motion arrays and diagnostics when requested."""
    if debug_prefix is None:
        return

    np.savez(
        f"{debug_prefix}-iter_{ind_iter + 1}.npz",
        iter_costs=iter_costs,
        trans=trans,
        rotvecs=rotvecs,
    )
    with open(f"{debug_prefix}-dbg-iter_{ind_iter + 1}.pkl", "wb") as fp:
        pickle.dump(debug_returns, fp)


def _print_run_summary(verbose, run_name, niter, solver, n_beat):
    """Print a concise summary before entering an iteration loop."""
    if not verbose:
        return

    method_name = solver.name if solver is not None else "None"
    print(f"{run_name}: niter={niter}, method={method_name}, n_beat={n_beat}")


def _print_iteration_setup(
    verbose,
    ind_iter,
    mode,
    solver,
    search_ub_tran,
    search_ub_rot,
    learning_rate,
    grid_spec,
):
    """Print the common configuration for one outer iteration."""
    if not verbose:
        return

    print(
        f"iteration={ind_iter}: mode={mode}, learning_rate={learning_rate}, "
        f"init_grid_spec={grid_spec}"
    )
    if solver is not None:
        print(f"search_ub_tran={search_ub_tran}")
        if mode == "rigid":
            print(f"search_ub_rot_deg={np.rad2deg(search_ub_rot)}")


def _print_partition_summary(
    verbose,
    ind_iter,
    partitions,
    inner_step=None,
    mode=None,
):
    """Print group-size information for beat partitions."""
    if not verbose:
        return

    location = f"iteration={ind_iter}"
    if inner_step is not None:
        location += f", inner_step={inner_step}"
    if mode is not None:
        location += f", mode={mode}"

    for partition_index, partition in enumerate(partitions):
        group_sizes = [len(group) for group in partition.groups]
        print(
            f"{location}, partition={partition_index}: "
            f"n_group={len(partition.groups)}, "
            f"min_group_size={min(group_sizes)}, "
            f"max_group_size={max(group_sizes)}"
        )


def _print_iteration_result(verbose, ind_iter, cost, elapsed_time):
    """Print the cost and elapsed time for a completed outer iteration."""
    if not verbose:
        return

    print(f"iteration={ind_iter}: cost={cost}, elapsed_time={elapsed_time} seconds")


def react(
    data_baseline,
    ks_baseline,
    ws,
    *,
    trans,
    rotvecs,
    niter,
    search_ub_tran,
    search_ub_rot,
    entropy_evaluator,
    deviator,
    reconstructor,
    solver,
    init_grid_schedule=None,
    partitioner=None,
    modes=None,
    func_learning_rate=None,
    bg_recompute_interval=0,
    rand_seed=42,
    verbose=False,
    debug_prefix=None,
):
    """Estimate motion by updating groups of beats to reduce image entropy.

    Parameters
    ----------
    data_baseline : ndarray
        Uncorrected k-space data, shape (ncoil, n_beat, ...).
    ks_baseline : ndarray
        Uncorrected sampling coordinates, as required by the motion model.
    ws : ndarray
        Required weights aligned with the sampling coordinates.
    trans, rotvecs : array_like
        Initial translations in physical units and rotations in radians, one
        row per beat. Component counts depend on the motion model. Input
        arrays may be modified.
    niter : int
        Number of iterations.
    search_ub_tran, search_ub_rot : array_like
        Search radii per iteration and motion component, in physical units
        for translations and radians for rotations.
    entropy_evaluator : BaseEntropyEvaluator
        Image-quality objective.
    deviator : BaseDeviator
        Motion model used to update data and sampling coordinates.
    reconstructor : BaseReconstructor
        Image reconstructor with subset updates.
    solver : BaseSolver or None
        Local optimizer. None uses the initial guess without optimization.
    init_grid_schedule : sequence of dict or None, optional
        Initial translation grids; see resolve_init_grid_param. None
        disables grid search.
    partitioner : BaseBeatPartitioner or None, optional
        Beat grouping strategy. None updates individual beats in random
        order.
    modes : {"rigid", "tran"} or sequence of str or None, optional
        One mode for all iterations, or one per iteration. None uses the
        deviator's current mode.
    func_learning_rate : callable or None, optional
        Maps an iteration index to a learning rate. None uses 1.
    bg_recompute_interval : int, optional
        Rebuild the background every this many group updates. Zero uses
        incremental updates only.
    rand_seed : int, optional
        Seed for grouping and update order.
    verbose : bool, optional
        Print progress.
    debug_prefix : str or None, optional
        Prefix for motion and diagnostic files. None disables saving.

    Returns
    -------
    trans, rotvecs : ndarray
        Estimated motion, with the same shapes and units as the inputs.
    iter_costs : ndarray, shape (niter + 1,)
        Objective values before and after each iteration.
    debug_returns : dict
        Motion, grouping, solver, and timing diagnostics.
    """
    if ws is None:
        raise ValueError("ws must be provided as an array, not None.")

    if modes is None:
        modes = [deviator.mode] * niter
    elif isinstance(modes, str):
        modes = [modes] * niter
    else:
        modes = list(modes)

    if len(modes) != niter:
        raise ValueError(f"Expected {niter} modes, got {len(modes)}.")
    if any(mode not in ("tran", "rigid") for mode in modes):
        raise ValueError("Modes must be 'tran' or 'rigid'.")

    rng = np.random.default_rng(rand_seed)
    trans = np.asarray(trans, dtype="float32")
    rotvecs = np.asarray(rotvecs, dtype="float32")

    max_mode = "rigid" if "rigid" in modes else "tran"
    deviator.set_mode(max_mode)

    n_beat = trans.shape[0]

    # note on shapes:
    # data: (ncoil, n_beat, n_intv, npt) (or ncoil, n_beat, ..., npt)
    # ks: (n_beat, n_intv, npt, ndim) (or n_beat, ..., npt, ndim)
    # ws: (n_beat, n_intv, npt) (or n_beat, ..., npt)

    # --- apply motion correction using initial estimation --- #

    data = np.zeros_like(data_baseline)
    ks = np.zeros_like(ks_baseline)
    for i in range(n_beat):
        ks[i] = deviator.motion_op.rotate_k(
            ks_baseline[i], deviator.motion_op.rotvec_to_rotmat(rotvecs[i])
        )
        data[:, i] = deviator.motion_op.apply_translation(
            data_baseline[:, i], ks_baseline[i], trans[i]
        )

    # --- prepare REACT --- #

    iter_costs = np.zeros(niter + 1, dtype="float32")
    debug_returns = _prepare_debug_returns(
        niter,
        trans,
        rotvecs,
        modes,
        partitioner,
    )

    # measure cost

    reconstructor.prep(data, ks, ws)
    recon = reconstructor.recon()
    iter_costs[0] = entropy_evaluator.eval(recon)

    _print_run_summary(verbose, "REACT", niter, solver, n_beat)

    for ind_iter in range(niter):
        start_time = time.perf_counter()
        deviator.set_mode(modes[ind_iter])

        grid_spec = None if init_grid_schedule is None else init_grid_schedule[ind_iter]
        init_grid_param = resolve_init_grid_param(grid_spec, trans)

        # --- determine learning rate --- #

        if func_learning_rate is None:
            learning_rate = 1.0
        else:
            learning_rate = float(func_learning_rate(ind_iter))

        _print_iteration_setup(
            verbose,
            ind_iter,
            deviator.mode,
            solver,
            search_ub_tran[ind_iter],
            search_ub_rot[ind_iter],
            learning_rate,
            grid_spec,
        )

        # ---- partition beats into update groups ---- #

        if partitioner is not None:
            partition_step = partitioner.make_partitions(
                trans,
                rotvecs,
                ind_iter=ind_iter,
                n_beat=n_beat,
                rng=rng,
            )
        else:
            groups = [[i] for i in rng.permutation(n_beat)]
            partition_step = PartitionStep(
                partitions=[
                    BeatPartition(
                        groups=groups,
                        group_windows=[
                            np.ones(len(group), dtype="float32") for group in groups
                        ],
                        ref_group_index=None,
                    )
                ],
                take_mean_partitions=False,
                freeze_reference_group=False,
            )

        partitions = partition_step.partitions
        _print_partition_summary(verbose, ind_iter, partitions)

        # --- one iteration of REACT --- #

        _, _, debug_return = react_step(
            trans,
            rotvecs,
            data,
            ks,
            partition_step=partition_step,
            data_baseline=data_baseline,
            ks_baseline=ks_baseline,
            ws=ws,
            entropy_evaluator=entropy_evaluator,
            deviator=deviator,
            reconstructor=reconstructor,
            solver=solver,
            search_ub_tran=search_ub_tran[ind_iter],
            search_ub_rot=search_ub_rot[ind_iter],
            init_grid_param=init_grid_param,
            learning_rate=learning_rate,
            bg_recompute_interval=bg_recompute_interval,
        )

        reconstructor.prep(data, ks, ws)
        recon = reconstructor.recon()
        iter_costs[ind_iter + 1] = entropy_evaluator.eval(recon)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # ---- save results after each iteration ---- #
        # WARNING: Debug only; compensate saved trans if roi_tran is set.
        # checkout "compensate for roi center shift" block in rigid3d.py.

        # save debug info

        _record_debug_iteration(
            debug_returns,
            ind_iter,
            iter_costs,
            trans,
            rotvecs,
            elapsed_time,
            partitions,
            debug_return,
        )
        _print_iteration_result(
            verbose,
            ind_iter,
            iter_costs[ind_iter + 1],
            elapsed_time,
        )

        _save_debug_outputs(
            debug_prefix,
            ind_iter,
            iter_costs,
            trans,
            rotvecs,
            debug_returns,
        )

    return (trans, rotvecs, iter_costs, debug_returns)


def sn_react(
    data_baseline,
    ks_baseline,
    ws,
    *,
    trans,
    rotvecs,
    niter,
    search_ub_tran,
    search_ub_rot,
    entropy_evaluator,
    deviator,
    reconstructor,
    solver,
    sn_param,
    init_grid_schedule=None,
    partitioner=None,
    modes=None,
    func_learning_rate=None,
    bg_recompute_interval=0,
    rand_seed=42,
    verbose=False,
    debug_prefix=None,
):
    """Estimate motion with groups informed by earlier motion updates.

    Parameters and return values follow ``react``, with these differences:

    - ``niter`` counts outer iterations. Each runs the partitioner's full
      grouping schedule; without a partitioner, it updates individual beats.
    - ``trans`` and ``rotvecs`` should start at zero.
    - ``sn_param`` is required. Set ``bootstrap_mode`` to "tran" or "rigid"
      (supported by the motion model), and ``delta_mode`` to "last" or
      "total" for the final inner update or the sum of inner updates.

    Optional ``sn_param`` entries are ``bootstrap_learning_rate`` (default
    1), ``bootstrap_wgt`` (fraction of bootstrap motion retained; default 0),
    and ``group_enabled_niter`` (outer iterations using grouping; default
    all). Motion-derived features start with an individual-beat bootstrap
    update and stay fixed within each outer iteration.
    """
    if ws is None:
        raise ValueError("ws must be provided as an array, not None.")

    if modes is None:
        modes = [deviator.mode] * niter
    elif isinstance(modes, str):
        modes = [modes] * niter
    else:
        modes = list(modes)

    if len(modes) != niter:
        raise ValueError(f"Expected {niter} modes, got {len(modes)}.")
    if any(mode not in ("tran", "rigid") for mode in modes):
        raise ValueError("Modes must be 'tran' or 'rigid'.")

    if sn_param is None:
        raise ValueError("sn_param is required for self-navigating REACT.")
    sn_param = dict(sn_param)
    bootstrap_learning_rate = float(sn_param.get("bootstrap_learning_rate", 1.0))
    bootstrap_wgt = float(sn_param.get("bootstrap_wgt", 0))
    bootstrap_mode = sn_param.get("bootstrap_mode")
    delta_mode = sn_param.get("delta_mode")
    group_enabled_niter = sn_param.get("group_enabled_niter")

    if bootstrap_mode not in ("tran", "rigid"):
        raise ValueError("bootstrap_mode must be 'tran' or 'rigid'.")
    if delta_mode not in ("last", "total"):
        raise ValueError("delta_mode must be 'last' or 'total'.")

    rng = np.random.default_rng(rand_seed)
    trans = np.asarray(trans, dtype="float32")
    rotvecs = np.asarray(rotvecs, dtype="float32")
    delta_trans = np.zeros_like(trans)
    delta_rotvecs = np.zeros_like(rotvecs)

    n_beat = trans.shape[0]
    feature_source = (
        None if partitioner is None else getattr(partitioner, "feature_source", None)
    )
    dynamic_feature_sources = ("motion", "delta", "motion_and_delta")

    def make_atomic_step():
        groups = [
            np.asarray([beat_index], dtype=int)
            for beat_index in rng.permutation(n_beat)
        ]
        return PartitionStep(
            partitions=[
                BeatPartition(
                    groups=groups,
                    group_windows=[
                        np.ones(len(group), dtype="float32") for group in groups
                    ],
                    ref_group_index=None,
                )
            ],
            take_mean_partitions=False,
            freeze_reference_group=False,
        )

    def make_feature_state(previous_delta_trans, previous_delta_rotvecs):
        if feature_source == "delta":
            return previous_delta_trans.copy(), previous_delta_rotvecs.copy()
        if feature_source == "motion_and_delta":
            return (
                np.concatenate((trans, previous_delta_trans), axis=1),
                np.concatenate((rotvecs, previous_delta_rotvecs), axis=1),
            )
        return trans.copy(), rotvecs.copy()

    # --- apply motion correction using the current estimates --- #

    data = np.zeros_like(data_baseline)
    ks = np.zeros_like(ks_baseline)
    for ind_beat in range(n_beat):
        ks[ind_beat] = deviator.motion_op.rotate_k(
            ks_baseline[ind_beat],
            deviator.motion_op.rotvec_to_rotmat(rotvecs[ind_beat]),
        )
        data[:, ind_beat] = deviator.motion_op.apply_translation(
            data_baseline[:, ind_beat],
            ks_baseline[ind_beat],
            trans[ind_beat],
        )

    # --- prepare diagnostics --- #

    iter_costs = np.zeros(niter + 1, dtype="float32")
    debug_returns = _prepare_debug_returns(
        niter,
        trans,
        rotvecs,
        modes,
        partitioner,
        extra_fields={
            "partition_steps": [[] for _ in range(niter)],
            "feature_source": feature_source,
            "group_enabled_niter": group_enabled_niter,
        },
    )

    reconstructor.prep(data, ks, ws)
    recon = reconstructor.recon()
    iter_costs[0] = entropy_evaluator.eval(recon)

    _print_run_summary(verbose, "SN-REACT", niter, solver, n_beat)

    for ind_outer in range(niter):
        start_time = time.perf_counter()
        deviator.set_mode(modes[ind_outer])

        grid_spec = (
            None if init_grid_schedule is None else init_grid_schedule[ind_outer]
        )
        init_grid_param = resolve_init_grid_param(grid_spec, trans)

        if func_learning_rate is None:
            learning_rate = 1.0
        else:
            learning_rate = float(func_learning_rate(ind_outer))

        _print_iteration_setup(
            verbose,
            ind_outer,
            deviator.mode,
            solver,
            search_ub_tran[ind_outer],
            search_ub_rot[ind_outer],
            learning_rate,
            grid_spec,
        )

        executed_partitions = []
        partition_steps = []

        grouping_enabled = partitioner is not None and (
            group_enabled_niter is None or ind_outer < group_enabled_niter
        )

        # --- prepare the frozen grouping feature and current outer delta --- #

        previous_delta_trans = delta_trans.copy()
        previous_delta_rotvecs = delta_rotvecs.copy()
        total_delta_trans = np.zeros_like(trans)
        total_delta_rotvecs = np.zeros_like(rotvecs)
        last_delta_trans = np.zeros_like(trans)
        last_delta_rotvecs = np.zeros_like(rotvecs)

        # --- bootstrap the first grouping feature with an atomic update --- #

        bootstrap_required = (
            grouping_enabled
            and ind_outer == 0
            and feature_source in dynamic_feature_sources
        )

        if bootstrap_required:
            partition_step = make_atomic_step()
            partitions = partition_step.partitions

            _print_partition_summary(
                verbose,
                ind_outer,
                partitions,
                inner_step="bootstrap",
                mode=bootstrap_mode,
            )

            outer_mode = deviator.mode
            deviator.set_mode(bootstrap_mode)

            (
                bootstrap_delta_trans,
                bootstrap_delta_rotvecs,
                step_debug,
            ) = react_step(
                trans=trans,
                rotvecs=rotvecs,
                data=data,
                ks=ks,
                partition_step=partition_step,
                data_baseline=data_baseline,
                ks_baseline=ks_baseline,
                ws=ws,
                entropy_evaluator=entropy_evaluator,
                deviator=deviator,
                reconstructor=reconstructor,
                solver=solver,
                search_ub_tran=search_ub_tran[ind_outer],
                search_ub_rot=search_ub_rot[ind_outer],
                init_grid_param=init_grid_param,
                learning_rate=bootstrap_learning_rate,
                bg_recompute_interval=bg_recompute_interval,
            )
            deviator.set_mode(outer_mode)
            executed_partitions.extend(partitions)
            partition_steps.extend(step_debug["partition_steps"])

            trans[:] = bootstrap_wgt * bootstrap_delta_trans
            rotvecs[:] = bootstrap_wgt * bootstrap_delta_rotvecs
            for ind_beat in range(n_beat):
                ks[ind_beat] = deviator.motion_op.rotate_k(
                    ks_baseline[ind_beat],
                    deviator.motion_op.rotvec_to_rotmat(rotvecs[ind_beat]),
                )
                data[:, ind_beat] = deviator.motion_op.apply_translation(
                    data_baseline[:, ind_beat],
                    ks_baseline[ind_beat],
                    trans[ind_beat],
                )
            reconstructor.prep(data, ks, ws)

            feature_trans = bootstrap_delta_trans.copy()
            feature_rotvecs = bootstrap_delta_rotvecs.copy()
        else:
            feature_trans, feature_rotvecs = make_feature_state(
                previous_delta_trans,
                previous_delta_rotvecs,
            )

        # --- execute this outer iteration's update schedule --- #

        inner_niter = partitioner.inner_niter if grouping_enabled else 1
        for ind_inner in range(inner_niter):
            if grouping_enabled:
                partition_step = partitioner.make_partitions(
                    feature_trans,
                    feature_rotvecs,
                    ind_iter=ind_inner,
                    n_beat=n_beat,
                    rng=rng,
                )
            else:
                partition_step = make_atomic_step()

            partitions = partition_step.partitions
            _print_partition_summary(
                verbose,
                ind_outer,
                partitions,
                inner_step=ind_inner,
            )

            inner_delta_trans, inner_delta_rotvecs, step_debug = react_step(
                trans=trans,
                rotvecs=rotvecs,
                data=data,
                ks=ks,
                partition_step=partition_step,
                data_baseline=data_baseline,
                ks_baseline=ks_baseline,
                ws=ws,
                entropy_evaluator=entropy_evaluator,
                deviator=deviator,
                reconstructor=reconstructor,
                solver=solver,
                search_ub_tran=search_ub_tran[ind_outer],
                search_ub_rot=search_ub_rot[ind_outer],
                init_grid_param=init_grid_param,
                learning_rate=learning_rate,
                bg_recompute_interval=bg_recompute_interval,
            )
            executed_partitions.extend(partitions)
            partition_steps.extend(step_debug["partition_steps"])

            last_delta_trans = inner_delta_trans.copy()
            last_delta_rotvecs = inner_delta_rotvecs.copy()
            total_delta_trans += inner_delta_trans
            for ind_beat in range(n_beat):
                total_delta_rotvecs[ind_beat] = deviator.motion_op.combine_rotvecs(
                    inner_delta_rotvecs[ind_beat],
                    total_delta_rotvecs[ind_beat],
                )

        if delta_mode == "last":
            delta_trans = last_delta_trans
            delta_rotvecs = last_delta_rotvecs
        else:
            delta_trans = total_delta_trans
            delta_rotvecs = total_delta_rotvecs

        reconstructor.prep(data, ks, ws)
        recon = reconstructor.recon()
        iter_costs[ind_outer + 1] = entropy_evaluator.eval(recon)

        elapsed_time = time.perf_counter() - start_time
        _record_debug_iteration(
            debug_returns,
            ind_outer,
            iter_costs,
            trans,
            rotvecs,
            elapsed_time,
            executed_partitions,
            {"partition_steps": partition_steps},
        )
        _print_iteration_result(
            verbose,
            ind_outer,
            iter_costs[ind_outer + 1],
            elapsed_time,
        )

        _save_debug_outputs(
            debug_prefix,
            ind_outer,
            iter_costs,
            trans,
            rotvecs,
            debug_returns,
        )

    return (trans, rotvecs, iter_costs, debug_returns)
