# TrimFields

TrimFields is a dataset of trimmed spline designs with isogeometric reference fields, made for neural solvers and
topology optimization on trimmed CAD faces. A design is a set of closed periodic cubic B-spline loops (trimming
curves) in the parameter square of a NURBS host patch, exported as an exact STEP face. This repository holds the
`trimfields` Python package and the notebook that generates the dataset.

**Dataset download:** [Download Dataset](https://drive.google.com/file/d/123a2Ya-3tpVU9ix4aQUIT-bgrXWSqWXj/view?usp=drive_link)
## Contents of the dataset

25,300 designs on 13 host families: 8 planar families (plane stress) and 5 shell families (Kirchhoff-Love), two
planar hosts and one shell host held out of training.

| Kind | Planar | Shells |
|---|---|---|
| Optimized (isogeometric SIMP, loop fit, CAD-native refinement) | 13,000 | 1,300 |
| Random admissible loop sets | 10,000 | 1,000 |

Every design stores its problem with continuum boundary conditions, its loops, the exported STEP face, the
displacement of a crisp cut-cell isogeometric evaluation, 64 samples per loop (position, normal, von Mises stress,
energy density) and its design fields on a 64 x 64 grid. Test designs also store the compliance of a finer
evaluation. Every design is reproducible from its index.

## Quick start

Linux x86_64, Python 3.10 to 3.12; an NVIDIA GPU is optional. The notebook installs its dependencies
(`requirements.txt`), runs the numerical checks and generates the dataset.

```bash
conda create -n trimfields python=3.11 -y && conda activate trimfields
pip install jupyter nbconvert ipykernel

# smoke test: every stage on one small design per family and split
TRIMFIELDS_MODE=smoke jupyter nbconvert --to notebook --execute data_generate.ipynb \
    --output data_generate.smoke.ipynb --ExecutePreprocessor.timeout=-1

# full dataset (about 280 core-hours, resumable)
jupyter nbconvert --to notebook --execute data_generate.ipynb \
    --output data_generate.run.ipynb --ExecutePreprocessor.timeout=-1
```

Outputs are written to `trimfields_data/` (`TRIMFIELDS_OUT`): `designs/` (one `.npz` per design), `step/` (one STEP
face per design), `consolidated/` (index tables, tensors, fields, splits) and `manifest.json`. After a change to the
package, `python build_notebook.py` rebuilds the notebook.

## Package

| Module | Content |
|---|---|
| `hosts`, `basis` | NURBS host patches of every family |
| `mechanics`, `simp` | Isogeometric plane stress and Kirchhoff-Love shells, SIMP |
| `loops`, `indicator`, `refine` | Loop fitting, differentiable material indicator (JAX), CAD-native refinement |
| `evaluate`, `fields`, `conditioning` | Crisp cut-cell evaluation, reference fields, conditioning fields |
| `problems`, `pipeline` | Problem sampling, one design end to end, resumable generation and consolidation |
| `export_step` | STEP export and read-back (OpenCASCADE) |
| `checks`, `bench`, `bodyfit` | Numerical checks, machine benchmark, optional body-fitted reference (gmsh) |

## License

MIT, see `LICENSE`.
