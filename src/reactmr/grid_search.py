import numpy as np


def compute_principal_directions(motion):
    """Return principal motion directions and their energy fractions.

    ``motion`` has shape (N, ndim), relative to a reference. Directions are
    orthonormal rows, shape (ndim, ndim), ordered by decreasing energy.
    """
    motion = np.asarray(motion)
    _, singular_values, dir_vectors = np.linalg.svd(motion, full_matrices=True)

    energies = np.zeros(motion.shape[1], dtype=singular_values.dtype)
    energies[: singular_values.size] = singular_values**2
    total_energy = energies.sum()
    fractions = energies / total_energy if total_energy > 0 else energies
    return dir_vectors, fractions


def parse_cartesian_directions(order, ndim, *, dtype=float):
    """Return unit direction vectors in the requested axis order.

    ``order`` is an axis string such as "xy" or "zyx". Output shape is
    (len(order), ndim), with the requested ``dtype``.
    """
    axis_indices = {"x": 0, "y": 1, "z": 2}
    indices = [axis_indices[axis] for axis in order]
    return np.eye(ndim, dtype=dtype)[indices]


def resolve_init_grid_param(grid_spec, motion):
    """Convert a translation-grid specification to numeric directions.

    ``grid_spec`` contains "dirs", "ranges", and "spacings". "dirs" is
    an axis string (for example, "xy") or a number of principal motion
    directions. Ranges and spacings follow that direction order and use
    the same physical units as ``motion``, which has shape (N, ndim).
    Returns a dictionary of numeric grid settings, or None when
    ``grid_spec`` is None or "dirs" is zero.
    """
    if grid_spec is None or grid_spec["dirs"] == 0:
        return None

    motion = np.asarray(motion)
    dir_spec = grid_spec["dirs"]
    if isinstance(dir_spec, str):
        grid_dirs = parse_cartesian_directions(
            dir_spec,
            motion.shape[-1],
            dtype=motion.dtype,
        )
    else:
        grid_dirs, _ = compute_principal_directions(motion)
        grid_dirs = grid_dirs[:dir_spec]

    return {
        "dirs": grid_dirs,
        "ranges": grid_spec["ranges"],
        "spacings": grid_spec["spacings"],
    }
