import numpy as np
import cupy as cp
from cupyx.scipy.sparse.linalg import LinearOperator

from .core import BaseReconstructor
from .utils import (
    to_device_array,
    from_device_array,
)

from mrgridding.kspace import SenseOperator


class Identity(LinearOperator):
    """Identity operator for a vector of length ``n``.

    ``is_complex`` selects complex64 or float32. ``gpu_id`` and ``stream``
    identify the CUDA resources.
    """

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


class SoSReconstructor(BaseReconstructor):
    """Reconstruct cropped sum-of-squares images from non-Cartesian data.

    ``ncoil`` is the coil count. ``sz_im`` and ``sz_crop`` are full and
    cropped image shapes in array order; ``roi_slices`` selects the crop.
    ``res`` gives pixel or voxel spacing in physical (x, y[, z]) order.
    ``ks`` has shape (n_beat, ..., ndim); required ``ws`` matches its samples.

    ``sens`` supplies coil sensitivity maps or None. ``osf`` and
    ``kernel_width`` control gridding. ``gpu_id`` and ``stream`` select the
    CUDA resources. ``denoise_param`` supplies SplitBregman settings;
    None disables denoising.
    """

    def __init__(
        self,
        *,
        sz_crop,
        roi_slices,
        ncoil,
        sz_im,
        res,
        ks,
        ws,
        sens,
        osf,
        kernel_width,
        gpu_id,
        stream,
        denoise_param=None,
    ):
        if ws is None:
            raise ValueError("ws must be provided as an array, not None.")
        self.ndim = len(sz_im)
        self.roi_slices = roi_slices
        self.ncoil = ncoil
        self.sz_im = sz_im
        self.sz_crop = sz_crop
        self.res = np.array(res, dtype="float32")
        self.gpu_id = gpu_id
        self.stream = stream
        self.model_full = SenseOperator(
            ncoil=ncoil,
            sz_im=sz_im,
            k=(self.res * ks.reshape(-1, self.ndim)),
            dcf=ws.ravel(),
            sens=sens,
            gpu_id=gpu_id,
            stream=stream,
            osf=osf,
            kernel_width=kernel_width,
            roi_slices=roi_slices,
        )
        self.model_subset = SenseOperator(
            ncoil=ncoil,
            sz_im=sz_im,
            k=(self.res * ks[:1].reshape(-1, self.ndim)),
            dcf=ws[:1].ravel(),
            sens=sens,
            gpu_id=gpu_id,
            stream=stream,
            osf=osf,
            kernel_width=kernel_width,
            roi_slices=roi_slices,
        )
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

    def prep(self, data, ks, ws):
        """Prepare the full background and clear the subset contribution.

        ``data`` has shape (ncoil, n_beat, ...), ``ks`` has shape
        (n_beat, ..., ndim), and required ``ws`` matches the trajectory samples.
        """
        if ws is None:
            raise ValueError("ws must be provided as an array, not None.")
        self.model_full.set_k((self.res * ks.reshape(-1, self.ndim)).astype("float32"))
        self.model_full.set_dcf(ws.ravel().astype("float32"))

        d_data = to_device_array(
            data.reshape(self.ncoil, -1), gpu_id=self.gpu_id, stream=self.stream
        )
        self.d_grids_bg = self.model_full.gridding_forward(d_data)
        with cp.cuda.Device(self.gpu_id), self.stream as stream:
            self.d_grids_subset = cp.zeros_like(self.d_grids_bg)
            stream.synchronize()

    def recon(self, *, to_host=False, for_scaling=False):
        """Return the sum-of-squares magnitude image with shape ``sz_crop``.

        ``to_host=True`` returns NumPy data; otherwise it returns CuPy data.
        ``for_scaling=True`` skips denoising.
        """
        if for_scaling:
            denoise = False
        else:
            denoise = True

        self.run_sum_of_squares(denoise=denoise)
        if to_host:
            return from_device_array(self.d_im, gpu_id=self.gpu_id, stream=self.stream)
        else:
            return self.d_im

    def subset_prep(self, data, ks, ws, *, mode="add"):
        """Add or subtract a subset's contribution to the background.

        Inputs follow ``prep`` but contain only the subset's beats.
        ``mode`` is "add" or "subtract"; ``ws`` is required.
        """
        self.subset_set_traj(ks, ws=ws)
        self.subset_gridding(data)
        with cp.cuda.Device(self.gpu_id), self.stream as stream:
            if mode == "add":
                self.d_grids_bg += self.d_grids_subset
            else:
                self.d_grids_bg -= self.d_grids_subset
            stream.synchronize()

    def subset_recon(self, data, ks, ws):
        """Return a CuPy magnitude image with the candidate subset included.

        Inputs follow ``prep`` but contain only the subset's beats. ``ks=None``
        reuses the current subset trajectory; ``ws`` is always required.
        Output shape is ``sz_crop``.
        """
        self.subset_set_traj(ks, ws=ws)
        self.subset_gridding(data)
        self.run_sum_of_squares()
        return self.d_im

    def subset_set_traj(self, ks=None, *, ws):
        """Set the subset trajectory and required weights ``ws``.

        ``ks`` has shape (n_subset, ..., ndim); None keeps the trajectory.
        ``ws`` must match the subset samples and cannot be None.
        """
        if ws is None:
            raise ValueError("ws must be provided as an array, not None.")
        if ks is not None:
            self.model_subset.set_k(
                (self.res * ks.reshape(-1, self.ndim)).astype("float32")
            )
        else:
            pass

        self.model_subset.set_dcf(ws.ravel().astype("float32"))

    def subset_gridding(self, data):
        """Grid subset data with shape (ncoil, n_subset, ...).

        Set the subset trajectory and weights before calling this method.
        """
        d_data = to_device_array(
            data.reshape(self.ncoil, -1), gpu_id=self.gpu_id, stream=self.stream
        )
        self.d_grids_subset = self.model_subset.gridding_forward(d_data)
        with cp.cuda.Device(self.gpu_id), self.stream as stream:
            stream.synchronize()

    def run_sum_of_squares(self, *, denoise=True):
        """Combine background and subset images into ``self.d_im``.

        Apply configured denoising unless ``denoise=False``.
        """
        with cp.cuda.Device(self.gpu_id), self.stream as stream:
            self.d_im = cp.sqrt(
                cp.sum(cp.abs(self.d_grids_bg + self.d_grids_subset) ** 2, axis=0)
            )
            stream.synchronize()

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
