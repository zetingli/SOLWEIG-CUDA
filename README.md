# SOLWEIG-CUDA

SOLWEIG-CUDA calculates mean radiant temperature (TMRT), UTCI and radiation
fields on an NVIDIA GPU using CuPy and CUDA kernels. It processes a regular
meteorological time series on one fixed raster domain, reusing geometry and
sky view factors and resetting thermal state each calendar day.
By default, a background worker transfers and writes results while
the GPU continues computing.

## Performance

Numerical validation against the standard SOLWEIG model shows excellent
agreement.

### Runtime

On the test workstation (Windows 11, Intel Core i9-13900K, 32 GB RAM and
NVIDIA GeForce RTX 4090 with 24 GB VRAM), a Dublin tile of **2000 × 2000
pixels at 1 m resolution** (2 × 2 km) required approximately:

| Simulation period | Hourly time steps | Complete workflow time |
| --- | ---: | ---: |
| 1 day | 24 | 2.7 s |
| 30 consecutive days | 720 | 21 s |

These rounded times represent means of five single-day runs and three
30-day runs, after one complete warm-up for each simulation period. Timing
includes input loading, fresh wall and SVF preparation, dynamic calculations, result
transfers and compressed GeoTIFF output, including the final wait for all
output to finish. Hourly MRT and UTCI are written using lossless DEFLATE
compression. Static results are computed once per run and reused across
days. Runtime depends on the scene, hardware and requested outputs.

### Reference implementations

Separate single-day benchmarks gave the following approximate speedups:

| Reference implementation | Version | SOLWEIG-CUDA speedup |
| --- | --- | ---: |
| Standard SOLWEIG in QGIS/UMEP | UMEP 4.1.1 | 4,000× |
| SOLWEIG-GPU | 2.0 | 100× |
| GLIDE-SOL | 0.1.1 | 50× |

These speedups are approximate and apply to the specific benchmark conditions.
The timings include each implementation's default output workflow, including
its native compression and file writing.

- Lindberg, F. et al. (2018).
  Urban Multi-scale Environmental Predictor (UMEP):
  An integrated tool for city-based climate services.
  Environmental Modelling & Software, 99, 70–87.
  [DOI](https://doi.org/10.1016/j.envsoft.2017.09.020).
- Kamath, H. G., Sudharsan, N., Singh, M., Wallenberg, N., Lindberg, F., and
  Niyogi, D. (2026). SOLWEIG-GPU: GPU-Accelerated Thermal Comfort Modeling
  Framework for Urban Digital Twins. *Journal of Open Source Software*,
  11(118), 9535. [DOI](https://doi.org/10.21105/joss.09535).
- Zonato, A., Kamath, H. G., Sudharsan, N., Monaco, L., Kittner, J., Wolf, L.,
  Demuzere, M., Middel, A., Bechtel, B., and Milelli, M. (2026). GLIDE-SOL: a
  GPU-accelerated global lightweight infrastructure for diagnostic
  environmental modeling with SOLWEIG. *Geoscientific Model Development*,
  19, 7389–7413. [DOI](https://doi.org/10.5194/gmd-19-7389-2026).

## Environment

```console
conda env create -f environment.yml
conda activate solweig-cuda
```

Simulation requires an NVIDIA GPU and a driver supporting CUDA 12. The
provided environment supplies the CUDA runtime and compiler. Windows with
Python 3.11 is tested; Linux has not been validated. Kernels compile on the
first run. There is no CPU simulation backend.

## Run

With the dependencies available, run from this source directory without
installing SOLWEIG-CUDA:

```console
python run.py --case examples/case.json
```
For a local case, prepare the referenced rasters and meteorology first.
Check the input before simulation.

```console
python scripts/input_check.py --case examples/case.json
python run.py --case examples/case.json
```

Run the input check once, and again if the inputs or parameters change.
For an `open_data` case it first prepares missing inputs, then checks
configuration, rasters and meteorology on the CPU.

Installing the project is optional. To make the `solweig-cuda` command
available outside this directory:

```console
python -m pip install . --no-deps --no-build-isolation
solweig-cuda --case examples/case.json
```

Authors: Zeting Li. Cite [CITATION.cff](CITATION.cff).
The software is distributed under [GPL-3.0-or-later](LICENSE).
See [NOTICE.md](NOTICE.md) for upstream copyright, source attribution and modifications.

Code is under `src/`: `series.py` runs the time series, 
`model.py` evaluates each CUDA step, and `kernels/` contains
the CUDA calculations. The root `scripts/` directory contains input preparation,
and checking tools; `tiling.py` remains in the package.
`umep_solweig.py` combines the retained UMEP calculations, with only two 
small adjustments to the solar-position helper. 

Archived release and citation: [Zenodo](https://zenodo.org/records/23328687).
