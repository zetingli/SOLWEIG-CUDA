# Source and licence notices

SOLWEIG-CUDA is distributed under the GNU General Public License, version 3
or any later version (`GPL-3.0-or-later`). See [LICENSE](LICENSE).

## SOLWEIG and UMEP

SOLWEIG-CUDA includes source code from SOLWEIG 2022a in the UMEP 4.1.1
distribution. The retained routines are consolidated in
[`src/umep_solweig.py`](src/umep_solweig.py). The model's spatial and radiation
calculations are also adapted for execution through custom CUDA kernels.
The upstream project is [UMEP](https://github.com/UMEP-dev/UMEP).

The original SOLWEIG package carries the following copyright notice:

> Copyright (C) 2016 by Fredrik Lindberg, UoG.

Its licence statement permits redistribution and modification under the
GNU General Public License, version 2 or, at the recipient's option, any
later version. The UMEP distribution also includes the GPL version 3 text.
SOLWEIG-CUDA is distributed under GPL version 3 or later, consistent with
that permission.

The upstream copyright notices and contributor credits apply to the
inherited code. Credits and references retained within individual routines,
including the solar-position helper, remain applicable.

## Modifications

The consolidation and adaptations for this release were made by Zeting Li
in October 2026. They include combining upstream modules, adjusting imports
and internal calls, and adapting spatial and radiation calculations for
CUDA execution. The retained meteorological solar-position helper also
extracts NumPy scalar values and applies the low-solar-altitude guard from
UMEP for Processing 2.0.35. The standalone workflow adds reusable spatial
preparation and asynchronous result transfer and output.

## Scientific references

- Lindberg, F., Holmer, B. and Thorsson, S. (2008). SOLWEIG 1.0 – Modelling
  spatial variations of 3D radiant fluxes and mean radiant temperature in
  complex urban settings. *International Journal of Biometeorology*, 52,
  697–713. [DOI](https://doi.org/10.1007/s00484-008-0162-7).
- Lindberg, F. et al. (2018). Urban Multi-scale Environmental Predictor
  (UMEP): An integrated tool for city-based climate services.
  *Environmental Modelling & Software*, 99, 70–87.
  [DOI](https://doi.org/10.1016/j.envsoft.2017.09.020).

[CITATION.cff](CITATION.cff) provides the citation metadata for SOLWEIG-CUDA.
The references above identify the underlying model and platform.

## Dependencies and input data

Third-party dependencies and input datasets retain their own licences.
The software licence does not grant rights to redistribute input data.
Dataset sources and licence information are listed in
[input preparation](docs/preparation.md).
