import numpy as np
import nlopt
from .core import BaseSolver


class NloptBOBYQASolver(BaseSolver):
    """Bounded, derivative-free optimization using NLopt BOBYQA.

    Parameters
    ----------
    xtol_abs : float, optional
        Absolute parameter tolerance.
    ftol_abs : float or None, optional
        Absolute objective tolerance. None leaves it unset.
    maxeval : int or None, optional
        Maximum objective evaluations. None leaves it unset.
    initial_step : float or ndarray or None, optional
        Initial step size. None lets NLopt choose.
    fallback_on_roundoff : bool, optional
        Return the best evaluated point if roundoff stops optimization.
    """

    def __init__(
        self,
        *,
        xtol_abs=1e-3,
        ftol_abs=None,
        maxeval=None,
        initial_step=None,
        fallback_on_roundoff=True,
    ):
        super().__init__(name="BOBYQA")
        self.xtol_abs = xtol_abs
        self.ftol_abs = ftol_abs
        self.maxeval = maxeval
        self.initial_step = initial_step
        self.fallback_on_roundoff = fallback_on_roundoff

    def solve(self, fun, x0, *, lower_bounds, upper_bounds):
        """Minimize ``fun(x)`` within the bounds, starting at ``x0``.

        Bounds must match ``x0``. Returns the solution array and solver
        diagnostics.
        """
        x0 = np.asarray(x0)
        out_dtype = x0.dtype
        x0 = x0.astype(np.float64, copy=False)
        lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
        upper_bounds = np.asarray(upper_bounds, dtype=np.float64)
        n_dim = x0.size

        # Track best objective/point explicitly so we can recover from
        # roundoff-limited termination.
        best = {"x": x0.copy(), "f": np.inf}

        def objective(x, grad):
            # grad is ignored for LN_BOBYQA
            x_arr = np.asarray(x, dtype=np.float64)
            val = float(fun(x_arr))
            if np.isfinite(val) and (val < best["f"]):
                best["f"] = val
                best["x"] = x_arr.copy()
            if not np.isfinite(val):
                return np.finfo(np.float64).max
            return val

        opt = nlopt.opt(nlopt.LN_BOBYQA, n_dim)
        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)
        opt.set_min_objective(objective)
        opt.set_xtol_abs(float(self.xtol_abs))
        if self.ftol_abs is not None:
            opt.set_ftol_abs(float(self.ftol_abs))
        if self.maxeval is not None:
            opt.set_maxeval(int(self.maxeval))
        if self.initial_step is not None:
            if np.isscalar(self.initial_step):
                step = np.full(n_dim, float(self.initial_step), dtype=np.float64)
            else:
                step = np.asarray(self.initial_step, dtype=np.float64)
            opt.set_initial_step(step)

        status = "success"
        try:
            x_opt = opt.optimize(x0)
        except nlopt.RoundoffLimited:
            if not self.fallback_on_roundoff:
                raise
            x_opt = best["x"].copy()
            status = "roundoff_limited"

        info = {
            "nevals": opt.get_numevals(),
            "status": status,
            "result_code": opt.last_optimize_result(),
        }
        if np.isfinite(best["f"]):
            info["best_objective"] = float(best["f"])
        return x_opt.astype(out_dtype, copy=False), info
