import numpy as np
import cupy as cp
from cupyx.scipy import fftpack as cufftpack
from cupyx.scipy import fft as cufft


class CupyCenteredFFTN:
    """Centered n-D FFT using CuPy with a precomputed plan."""

    def __init__(self, sz_im, *, gpu_id=0, stream=None, dtype=np.complex64):
        """Create an FFT plan for image shape ``sz_im`` and complex ``dtype``.

        ``gpu_id`` selects the CUDA device. ``stream=None`` uses its null stream.
        """
        self.sz_im = tuple(sz_im)
        self.gpu_id = int(gpu_id)
        self.dtype = dtype

        if stream is None:
            stream = cp.cuda.Stream(null=True)
        self.stream = stream

        with cp.cuda.Device(self.gpu_id), self.stream:
            self.d_im = cp.empty(self.sz_im, dtype=self.dtype)
            self.d_plan = cufftpack.get_fft_plan(self.d_im)

    def destroy(self):
        """Explicit resource cleanup (optional)."""
        with cp.cuda.Device(self.gpu_id), self.stream:
            self.d_plan = None
            self.d_im = None

    def cfftn(self, im):
        """Return the centered n-D FFT of ``im`` using the GPU.

        NumPy input produces NumPy output; CuPy input produces CuPy output.
        """
        xp = cp.get_array_module(im)

        # copy to GPU if needed

        with cp.cuda.Device(self.gpu_id), self.stream:
            if xp is np:
                d_im = cp.asarray(im, dtype=self.dtype)
            else:
                d_im = im.astype(self.dtype, copy=False)

        # Centered FFTn: ifftshift -> fftn -> fftshift

        with cp.cuda.Device(self.gpu_id), self.stream:
            self.d_im = cufft.ifftshift(d_im)
            cufft.fftn(self.d_im, plan=self.d_plan, overwrite_x=True)
            d_k = cufft.fftshift(self.d_im)

        # Return

        with cp.cuda.Device(self.gpu_id), self.stream:
            if xp is np:
                return cp.asnumpy(d_k)
            else:
                return d_k

    def cifftn(self, im):
        """Return the centered inverse n-D FFT of ``im`` using the GPU.

        NumPy input produces NumPy output; CuPy input produces CuPy output.
        """
        xp = cp.get_array_module(im)

        # copy to GPU if needed

        with cp.cuda.Device(self.gpu_id), self.stream:
            if xp is np:
                d_im = cp.asarray(im, dtype=self.dtype)
            else:
                d_im = im.astype(self.dtype, copy=False)

        # Centered inverse FFTn: ifftshift -> ifftn -> fftshift

        with cp.cuda.Device(self.gpu_id), self.stream:
            self.d_im = cufft.ifftshift(d_im)
            cufft.ifftn(self.d_im, plan=self.d_plan, overwrite_x=True)
            d_k = cufft.fftshift(self.d_im)

        # Return

        with cp.cuda.Device(self.gpu_id), self.stream:
            if xp is np:
                return cp.asnumpy(d_k)
            else:
                return d_k
