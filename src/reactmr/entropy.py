import numpy as np
import cupy as cp

from .core import BaseEntropyEvaluator
from .utils import to_device_array

from gradops.finite_diff_mag import FiniteDifferenceMagOp


class EntropyEvaluator(BaseEntropyEvaluator):
    """Evaluate image-gradient entropy within an optional mask.

    ``mask_crop`` selects pixels to evaluate; None uses the full crop.
    ``sz_crop`` is the image shape in array order. ``is_complex`` specifies
    whether the image is complex. ``gpu_id`` and ``stream`` select the CUDA
    resources. ``eval(im)`` returns a scalar cost.
    """

    def __init__(self, mask_crop, *, sz_crop, is_complex, gpu_id, stream):

        self.gpu_id = gpu_id
        self.stream = stream

        if mask_crop is not None:
            self.h_mask = mask_crop.copy()
            self.d_mask = to_device_array(mask_crop, gpu_id=gpu_id, stream=stream)
        else:
            self.h_mask = None
            self.d_mask = None

        self.grad_mag_op = FiniteDifferenceMagOp(
            sz_crop, len(sz_crop), is_complex, gpu_id, stream
        )

    def eval(self, im):

        mag = self.grad_mag_op.run(im)  # image gradient magnitude (H)

        xp = cp.get_array_module(im)
        if xp == np:
            if self.h_mask is not None:
                mag *= self.h_mask

            h_p = mag / (np.sum(mag) + 1e-19)
            h_entropy = -np.sum(h_p * np.log2(h_p + 1e-19))
            entropy = float(h_entropy)
        else:
            with cp.cuda.Device(self.gpu_id), self.stream as stream:
                if self.d_mask is not None:
                    mag *= self.d_mask

                d_p = mag / (cp.sum(mag) + 1e-19)
                d_entropy = -cp.sum(d_p * cp.log2(d_p + 1e-19))
                entropy = float(d_entropy)
                stream.synchronize()

        return entropy
