# REACT for MR Imaging

A Python implementation of Rigid Motion Estimation using Accelerated Coordinate Descent (REACT) for MR imaging.

## Requirements and installation

- Python 3.11 is suggested; package metadata currently allows Python >= 3.10.
- NumPy, SciPy, NLopt, and scikit-learn.
- A CUDA-capable GPU and a matching CuPy installation.
- [gradops](https://github.com/kejang/gradops) for image-gradient entropy.
- [splitbregman](https://github.com/kejang/splitbregman) only when denoising is enabled.
- `rigid2d` and `rigid3d` also require `mrgridding`, which is not yet publicly available. A release is planned soon at [kejang/mrgridding](https://github.com/kejang/mrgridding).

Install CuPy for your CUDA version, then:

```bash
pip install numpy scipy nlopt scikit-learn
pip install git+https://github.com/kejang/gradops.git
pip install git+https://github.com/kejang/reactmr.git
```

Denoising is disabled by default (`denoise_param=None`); `niter_outer=0` also disables it. To enable denoising, install `splitbregman`:

```bash
pip install git+https://github.com/kejang/splitbregman.git
```

## Demo

See [simul-tran2d-epi.ipynb](demo/simul-tran2d-epi.ipynb) for a 2D EPI translation example. As written, it uses `splitbregman`, Jupyter, Matplotlib, and SciencePlots. This demo does not need `mrgridding`.

## Usage

Choose a motion model and call its `run_react()` function:

- `reactmr.tran2d`: 2D translation for Cartesian EPI data.
- `reactmr.rigid2d`: 2D translation and rotation.
- `reactmr.rigid3d`: 3D translation and rotation.

Pass k-space data and sampling coordinates, plus density compensation weights for rigid motion. Set image size, resolution, iteration count, and search bounds by keyword; see each wrapper's docstring for details. Initial motion defaults to zero.

The function returns translations, rotation vectors, iteration costs, and diagnostics. `verbose=True` prints progress; `debug_prefix` saves motion and diagnostics to files.

## Conventions

- Images use array order `(z, y, x)` in 3D and `(y, x)` in 2D. Motion vectors and rigid-motion trajectories use `(x, y, z)` or `(x, y)`.
- Positive physical directions are x: right to left, y: anterior to posterior, and z: inferior to superior.
- Image sizes and `spacing` use array order; `res` uses physical axis order. Translation units match the resolution units. Rigid-motion trajectories use cycles per unit; `tran2d` takes ky row indices.
- Rotations are in radians: a right-hand rotation vector in 3D, or a single angle about +z in 2D.

Returned motion describes the **correction** relative to the reconstruction's reference pose. For pure translation, an object displacement of +1 along x gives a correcting translation of -1. Likewise, a rotation of +θ about an axis is corrected by -θ about that axis.

Specifically, translation `t` multiplies the data by `exp(-2πi k·t)` using the original trajectory, while rotation `R` changes the trajectory to `R @ k` (column-vector notation). With rotation present, `t` is expressed in the original acquisition axes: the combined spatial correction is `x_corrected = R @ (x + t)`.

## License

Sustainable Use License from [n8n](https://github.com/n8n-io/n8n). See [LICENSE](LICENSE).
