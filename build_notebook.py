"""Build data_generate.ipynb (default name; pass another as the first argument): a self-contained notebook that installs the dependencies, writes the trimfields
package, validates it, benchmarks the machine and generates, consolidates and packages the TrimFields dataset."""
import pathlib
import sys

import nbformat as nbf

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE / "trimfields"
OUT = HERE / (sys.argv[1] if len(sys.argv) > 1 else "data_generate.ipynb")
MODULES = ["__init__", "basis", "hosts", "mechanics", "problems", "simp", "loops", "indicator", "refine",
           "evaluate", "conditioning", "fields", "export_step", "bodyfit", "pipeline", "checks", "bench"]

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


md(r"""
# TrimFields dataset generation

This notebook generates the TrimFields dataset: trimmed spline designs on NURBS hosts with isogeometric reference
fields, for neural solvers on trimmed CAD faces. Each design is stored as a trimmed surface (periodic cubic
trimming loops in the parameter square of the host) together with a STEP file whose trimming edges are pcurves on
the host surface, the displacement of the crisp cut-cell evaluation, stress and energy samples along every loop,
and the design fields (crisp indicator and signed distance) on the conditioning grid.

| Split | Families | Designs |
|---|---|---|
| `id` | optimized designs on planar hosts rect (rectangle), sector (annular sector), plate (plate with a hole), arc (curved beam), taper (tapered beam), bolted (bolted plate) (2,000 each; every 10th design is test) | 12,000 |
| `ood_load` | optimized, rect to taper with the load centred in the held-out band of edge v1 (100 each) | 500 |
| `ood_host` | optimized, freeform (freeform patch), ell (L-shaped domain) (250 each) | 500 |
| `shell` | optimized shells cyl (cylinder) 400, saddle 300, dome 300, torus (torus sector) 200 (every 10th design is test) | 1,200 |
| `shell_ood` | optimized, hood (car-hood patch) | 100 |
| `rand` | random admissible loop sets on rect to bolted (1,500 each; every 10th design is test) | 9,000 |
| `rand_host` | random loop sets on freeform and ell (500 each) | 1,000 |
| `rand_shell` | random loop sets on cyl to torus (200 each; every 10th design is test) | 800 |
| `rand_shell_ood` | random loop sets on hood | 200 |

Optimized designs run isogeometric SIMP (Heaviside projection for shells), a crisp cut-cell evaluation of the
thresholded density, loop extraction and periodic B-spline fitting, the CAD-native refinement with exact design
gradients (JAX), an admissibility test and the STEP export, as in TrimTO. Random designs skip the optimization;
loop sets that cut the load path (compliance above 50 times that of the untrimmed host) are rejected, and the host
compliance is stored for every design as `J_host`. All boundary conditions are stated for the continuum (bearing tractions on loaded bolt holes, rollers only on
straight axis-aligned edges), and the designs of the test splits also store the compliance of a finer evaluation
(four times the elements per direction for planar faces, three times for shells). The counts above are the
configuration of the paper (about 280 core-hours on one core per worker, measured on the 570-design sample); `pipeline.CONFIG["counts"]` can be raised later without changing
any existing design, since every seed derives from the index.
Problems without an admissible trimmed design are resampled with a new seed; every choice is seeded, so the
dataset is reproducible.

**How to run.** Copy this notebook to the GPU machine (Linux x86_64, Python 3.10 to 3.12, NVIDIA driver 525 or
newer for the CUDA 12 build of JAX). The full run takes days, so a headless run inside `tmux` is recommended:

```bash
tmux new -s trimfields
jupyter nbconvert --to notebook --execute data_generate.ipynb \
    --output data_generate.run.ipynb --ExecutePreprocessor.timeout=-1
```

Running all cells in Jupyter works as well. The run is resumable: after an interruption, run the notebook again and
finished designs are skipped. Options are read from environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `TRIMFIELDS_MODE` | `full` | `full` generates the dataset; `smoke` runs every stage on one small design per family |
| `TRIMFIELDS_OUT` | `trimfields_data` | output directory |
| `TRIMFIELDS_WORKERS` | `auto` | worker processes (default: CPU cores minus one, limited by memory) |
| `TRIMFIELDS_GPU_WORKERS` | `auto` | workers that run the JAX part on the GPU (default: from the benchmark in Section 6) |
| `TRIMFIELDS_INSTALL` | `1` | `0` skips the pip installation |

**Outputs** (under `TRIMFIELDS_OUT`): `designs/<split>/<family>/<idx>.npz` (problem, metrics, SIMP density, fitted,
refined and final loops, canonical tensor, refinement history, evaluation-mesh displacement `u_eval`, loop
samples `loop_*`, design field), `step/<split>/<family>/<idx>.step`, `consolidated/` (index tables, design
tensors, conditioning and design fields, splits, quality report, gallery), and `manifest.json`; the archives with
their SHA-256 sums go to `<TRIMFIELDS_OUT>_archives/`.
""")

md("## 1. Parameters")
code(r"""
import copy
import importlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time

MODE = os.environ.get("TRIMFIELDS_MODE", "full").lower()              # "full" or "smoke"
OUT_DIR = os.path.abspath(os.environ.get("TRIMFIELDS_OUT", "trimfields_data"))
WORKERS = os.environ.get("TRIMFIELDS_WORKERS", "auto")                 # total worker processes
GPU_WORKERS = os.environ.get("TRIMFIELDS_GPU_WORKERS", "auto")         # workers running the JAX part on the GPU
INSTALL = os.environ.get("TRIMFIELDS_INSTALL", "1") == "1"
WORK = pathlib.Path.cwd()
assert MODE in ("full", "smoke"), f"TRIMFIELDS_MODE must be 'full' or 'smoke', not {MODE!r}"
T_START = time.monotonic()
print(f"mode {MODE} | output {OUT_DIR} | working directory {WORK} | Python {sys.version.split()[0]}")
""")

md("""
## 2. Dependencies

Python packages are installed into the running interpreter. JAX is installed with its CUDA 12 wheels when an
NVIDIA GPU is visible, and PARDISO (`pypardiso`, Intel MKL) replaces SuperLU as the sparse direct solver on x86_64.
The second cell imports every package in a fresh process and reinstalls any compiled package that was built
against another NumPy version (a common problem when a system Python mixes packages from several sources).
""")
code(r'''
ENV_HELP = """
Use a dedicated Python environment for this notebook, for example
  conda create -n trimfields python=3.11 -y && conda activate trimfields
or (Ubuntu: sudo apt install python3-venv)
  python3 -m venv ~/trimfields-env && source ~/trimfields-env/bin/activate
then
  pip install jupyter nbconvert ipykernel
and start Jupyter (or jupyter nbconvert) from that environment."""


def sh(cmd, check=True, **kw):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, **kw)


def pip_install(*pkgs, check=True):
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade-strategy", "only-if-needed", *pkgs]
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if out:
        print(out[-3000:])
    if r.returncode != 0:
        if "externally-managed-environment" in out:
            raise RuntimeError("pip may not install into this system Python." + ENV_HELP)
        if check:
            raise subprocess.CalledProcessError(r.returncode, cmd)
    return r.returncode == 0


def nvidia_gpus():
    if shutil.which("nvidia-smi") is None:
        return []
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return [l.strip() for l in r.stdout.splitlines() if l.strip()] if r.returncode == 0 else []


def dist_versions(names):
    import importlib.metadata as md
    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = None
    return out


GPUS = nvidia_gpus()
# Minimum versions are the first releases built against NumPy 2 (older compiled wheels fail to import under it).
REQUIREMENTS = ["numpy>=1.26", "scipy>=1.13", "matplotlib>=3.9", "contourpy>=1.2.1", "shapely>=2.0.4",
                "scikit-image>=0.24", "pandas>=2.2.2", "tqdm>=4.66", "cadquery-ocp>=7.7"]
TRACKED = ["numpy", "scipy", "matplotlib", "contourpy", "shapely", "scikit-image", "pandas", "jax", "jaxlib"]
BEFORE = dist_versions(TRACKED)
if INSTALL:
    assert sys.version_info[:2] >= (3, 10), "Python 3.10 or newer is required"
    pip_install("--upgrade", "pip")
    pip_install(*REQUIREMENTS)
    pip_install("jax[cuda12]>=0.4.30" if GPUS else "jax>=0.4.30")
    if platform.machine() in ("x86_64", "AMD64"):
        if not pip_install("pypardiso>=0.4.6", check=False):
            print("pypardiso could not be installed: SuperLU is used (about 3 times slower solves)")
print("Python:", sys.executable, sys.version.split()[0], "| GPUs:", GPUS or "none")
''')
code(r'''
# Every module the pipeline needs must import in a fresh process. A compiled package built against another NumPy
# ABI (for example an old matplotlib in ~/.local next to NumPy 2) is reinstalled from its current wheel.
IMPORTS = {"numpy": "numpy", "scipy": "scipy.sparse.linalg,scipy.spatial", "matplotlib": "matplotlib.path",
           "contourpy": "contourpy", "shapely": "shapely", "scikit-image": "skimage.measure", "pandas": "pandas",
           "tqdm": "tqdm", "jax": "jax.numpy"}
if platform.machine() in ("x86_64", "AMD64"):
    IMPORTS["pypardiso"] = "pypardiso"
PROBE = r"""
import importlib, json, os, sys, traceback
try:
    for m in sys.argv[1].split(","):
        importlib.import_module(m)
    print("IMPORT_OK")
except BaseException as e:
    files = [f.filename for f in traceback.extract_tb(e.__traceback__)]
    pk = []
    for f in files:
        for key in ("site-packages", "dist-packages"):
            if os.sep + key + os.sep in f:
                pk.append(f.split(os.sep + key + os.sep)[-1].split(os.sep)[0])
    print("IMPORT_FAIL " + json.dumps({"error": f"{type(e).__name__}: {e}", "packages": pk}))
"""


def import_failures():
    import re
    fails = {}
    for dist, mods in IMPORTS.items():
        try:
            r = subprocess.run([sys.executable, "-c", PROBE, mods], capture_output=True, text=True,
                               env=dict(os.environ, JAX_PLATFORMS="cpu"), timeout=600)
        except subprocess.TimeoutExpired:
            fails[dist] = {"error": "import timed out", "packages": [], "abi": False}
            continue
        abi = "_ARRAY_API not found" in r.stderr or "compiled using NumPy 1.x" in r.stderr
        line = [l for l in r.stdout.splitlines() if l.startswith("IMPORT_")]
        if line and line[-1].startswith("IMPORT_FAIL"):
            info = json.loads(line[-1][len("IMPORT_FAIL "):])
        elif not line:
            info = {"error": (r.stderr.strip().splitlines() or ["no output"])[-1], "packages": []}
        elif abi:
            info = {"error": "built against another NumPy version", "packages": []}
        else:
            continue
        if abi:                         # NumPy prints the importing stack: its innermost package is the culprit
            frames = re.findall(r"(?:site|dist)-packages[/\\\\]([A-Za-z0-9_.]+)", r.stderr)
            info["packages"] = info["packages"] + frames[-1:]
        info["abi"] = abi
        fails[dist] = info
    return fails


def owners(pkg_dirs):
    """Distributions that own the given top-level package directories."""
    import importlib.metadata as md
    top = md.packages_distributions()
    return {d for p in pkg_dirs for d in top.get(p[:-3] if p.endswith(".py") else p, [])}


for _ in range(3):
    FAILS = import_failures()
    if not FAILS:
        break
    for dist, info in FAILS.items():
        print(f"import of {dist} failed: {info['error'][:300]}")
    if not INSTALL:
        break
    fix = (set(FAILS) | owners(p for info in FAILS.values() for p in info["packages"])) - {"numpy"}
    print("reinstalling from current wheels:", sorted(fix))
    pip_install("--upgrade", "--force-reinstall", "--no-deps", *sorted(fix), check=False)
if FAILS:
    raise RuntimeError(f"these packages cannot be imported: {sorted(FAILS)}." + ENV_HELP)
AFTER = dist_versions(TRACKED)
changed = [d for d in TRACKED if BEFORE[d] != AFTER[d]]
stale = [d for d in changed if d.replace("scikit-image", "skimage") in sys.modules]
if stale:
    raise RuntimeError(f"{stale} changed after this kernel had imported them: restart the kernel and run all cells "
                       "again (finished work is kept).")
print("all imports ok:", ", ".join(f"{d} {v}" for d, v in AFTER.items() if v))
''')
code(r"""
# OpenCASCADE (cadquery-ocp) links libGL and libX11, which headless servers often lack.
def ocp_error():
    r = subprocess.run([sys.executable, "-c", "import OCP.BRepAlgoAPI, OCP.STEPControl"], capture_output=True,
                       text=True)
    return None if r.returncode == 0 else ((r.stderr.strip().splitlines() or ["unknown error"])[-1])


err = ocp_error()
if err:
    print("OCP import failed:", err)
    APT = ["apt-get", "install", "-y", "--no-install-recommends", "libgl1", "libx11-6", "libxcb1", "libexpat1"]
    if shutil.which("apt-get"):
        if os.geteuid() == 0:
            sh(["apt-get", "update"], check=False)
            sh(APT, check=False)
        elif shutil.which("sudo") and subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0:
            sh(["sudo", "-n", "apt-get", "update"], check=False)
            sh(["sudo", "-n"] + APT, check=False)
    err = ocp_error()
    if err:
        raise RuntimeError("OpenCASCADE cannot be imported: " + err + "\nInstall the system libraries, e.g.\n"
                           "  sudo apt-get install -y libgl1 libx11-6 libxcb1 libexpat1\nand run this cell again.")
print("OpenCASCADE (OCP) import: ok")
""")

md("## 3. Machine and environment")
code(r"""
import numpy as np


def mem_gb(key="MemAvailable"):
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 2**20
    return float("nan")


def versions():
    import importlib.metadata as md
    out = {}
    for p in ("numpy", "scipy", "jax", "jaxlib", "shapely", "scikit-image", "matplotlib", "pandas", "cadquery-ocp",
              "pypardiso", "mkl", "tqdm"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


CPUS = len(os.sched_getaffinity(0))
os.makedirs(OUT_DIR, exist_ok=True)
MACHINE = {"host": platform.node(), "platform": platform.platform(), "cpu": platform.processor() or platform.machine(),
           "cores": CPUS, "mem_total_gb": round(mem_gb("MemTotal"), 1), "mem_available_gb": round(mem_gb(), 1),
           "disk_free_gb": round(shutil.disk_usage(OUT_DIR).free / 2**30, 1), "gpus": GPUS, "python": sys.version,
           "versions": versions()}
print(json.dumps(MACHINE, indent=1))


def jax_probe(backend):
    code_ = ("import jax, jax.numpy as jnp; jax.config.update('jax_enable_x64', True); "
             "x = jnp.ones((256, 256)); print(jax.default_backend(), jax.devices()[0], float((x @ x).sum()))")
    env = dict(os.environ, JAX_PLATFORMS=backend, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    try:
        r = subprocess.run([sys.executable, "-c", code_], capture_output=True, text=True, env=env, timeout=600)
    except subprocess.TimeoutExpired:
        return False, "no answer within 10 minutes"
    return r.returncode == 0 and r.stdout.strip().endswith("16777216.0"), (r.stdout + r.stderr).strip()[-400:]


JAX_GPU_OK = False
if GPUS:
    JAX_GPU_OK, msg = jax_probe("cuda")
    print("JAX on the GPU:", "ok" if JAX_GPU_OK else "FAILED, the CPU is used\n" + msg)
print("JAX on the CPU:", "ok" if jax_probe("cpu")[0] else "FAILED")
if MACHINE["disk_free_gb"] < 15 and MODE == "full":
    print("WARNING: the full dataset with STEP files and archives needs about 15 GB of free disk space")
""")

md("""
## 4. The `trimfields` package

The next cells write the package next to this notebook (`./trimfields/`). Worker processes import it from there.
""")
code('os.makedirs("trimfields", exist_ok=True)')
for name in MODULES:
    src = (PKG / f"{name}.py").read_text()
    cells.append(nbf.v4.new_code_cell(f"%%writefile trimfields/{name}.py\n" + src.rstrip("\n") + "\n"))

md("## 5. Configuration and numerical checks")
code(r"""
os.environ["JAX_PLATFORMS"] = "cpu"                   # the notebook process runs JAX on the CPU; workers choose
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
for k in [k for k in sys.modules if k == "trimfields" or k.startswith("trimfields.")]:
    del sys.modules[k]
importlib.invalidate_caches()
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))
import trimfields
from trimfields import pipeline as P
from trimfields.checks import run_all_checks

cfg = copy.deepcopy(P.CONFIG)
cfg["out_dir"] = OUT_DIR
TASKS = P.task_list(cfg)
N_PLANAR = sum(n for s in P.PLANAR_SPLITS for n in cfg["counts"].get(s, {}).values())
N_SHELL = sum(n for s in P.SHELL_SPLITS for n in cfg["counts"].get(s, {}).values())
N_RANDOM = sum(n for s in P.RANDOM_SPLITS for n in cfg["counts"].get(s, {}).values())
print(f"trimfields {trimfields.__version__}: {len(TASKS)} designs ({N_PLANAR} planar, {N_SHELL} shell; "
      f"{N_RANDOM} of them random)")
for split, fams in cfg["counts"].items():
    print(f"  {split:10s}", ", ".join(f"{f} {n}" for f, n in fams.items()))
""")
code(r"""
CHECKS = run_all_checks()
failed = [k for k, v in CHECKS.items() if not v["pass"]]
assert not failed, f"numerical checks failed: {failed}"
""")

md("""
## 6. Benchmark: JAX backend, worker count and run time

Timings run in fresh processes with the settings of a worker. First, the JAX part of a refinement step (indicator,
gradient and volume) and the sparse solve; then one complete planar and one complete shell design per backend,
which also give the peak memory of a worker. With `n_g` GPU workers and `n_c` CPU workers, the throughput in
refinement steps per second is modelled as `min(n_g / (t_solve + t_gpu), n_gpus / t_gpu) + n_c / (t_solve + t_cpu)`
(each GPU is time-shared between processes). The split that maximizes it within the CPU cores and the memory of
the machine is used.
""")
code(r"""
def run_bench(args, backend, timeout=7200):
    env = dict(os.environ, JAX_PLATFORMS=backend, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_PYTHON_CLIENT_ALLOCATOR="platform")
    if backend == "cpu":
        env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    try:
        r = subprocess.run([sys.executable, "-m", "trimfields.bench", *map(str, args)], cwd=WORK, env=env,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"benchmark {args} on {backend}: no result within {timeout} s")
        return None
    res = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    if r.returncode != 0 or not res:
        print(f"benchmark {args} on {backend} failed:\n", r.stdout[-1500:], r.stderr[-3000:])
        return None
    return json.loads(res[-1][7:])


N_EL = cfg["n_el_planar"] if MODE == "full" else 900
BACKENDS = ["cpu"] + (["cuda"] if JAX_GPU_OK else [])
BENCH = {b: run_bench(["indicator", N_EL, 5 if b == "cpu" else 20], b) for b in BACKENDS}
for b, r in BENCH.items():
    if r:
        print(f"{b:5s}: JAX step {r['t_indicator']*1e3:8.1f} ms | solve {r['t_solve']*1e3:6.1f} ms "
              f"(PARDISO {r['pardiso']}) | compile {r['t_compile']:.1f} s")
assert BENCH["cpu"], "the CPU benchmark failed"
if JAX_GPU_OK and not BENCH.get("cuda"):
    BACKENDS = ["cpu"]
""")
code(r"""
# Complete designs (all stages, including resampled attempts): wall time and peak memory per backend.
bench_dir = os.path.join(OUT_DIR, "_bench")
shutil.rmtree(bench_dir, ignore_errors=True)
small = [] if MODE == "full" else ["small"]
DESIGN_T, DESIGN_MEM = {}, {}
for backend in BACKENDS:
    for kind, (split, fam) in {"planar": ("id", "plate"), "shell": ("shell", "cyl")}.items():
        r = run_bench(["design", split, fam, 10, os.path.join(bench_dir, backend)] + small, backend)   # index 10: a test design, whose finer evaluation sets the memory peak
        if r is None or r["status"] != "ok":
            print(f"design benchmark {kind} on {backend} failed: {r}")
            continue
        DESIGN_T[(backend, kind)] = r["wall"]
        DESIGN_MEM[backend] = max(DESIGN_MEM.get(backend, 0.0), r["max_rss_gb"])
        print(f"{kind:6s} design on {backend:4s}: {r['wall']:6.1f} s, peak memory {r['max_rss_gb']:.2f} GB "
              f"(attempt {r['attempt']}, {r['nel']} elements; " + ", ".join(f"{k} {v:.1f}" for k, v in
                                                                          r["times"].items()) + " s)")
shutil.rmtree(bench_dir, ignore_errors=True)
if "cuda" in BACKENDS and not all(("cuda", k) in DESIGN_T for k in ("planar", "shell")):
    print("the GPU design benchmark failed: all workers run on the CPU")
    BACKENDS = ["cpu"]
assert all(("cpu", k) in DESIGN_T for k in ("planar", "shell")), "the CPU design benchmark failed"
""")
code(r"""
t_cpu, t_solve = BENCH["cpu"]["t_indicator"], BENCH["cpu"]["t_solve"]
t_rest = t_solve + 0.02                                       # solve plus the remaining NumPy work of a step
MEM = mem_gb() - 2.0                                          # keep 2 GB for the system and the notebook
m_cpu = 1.25 * DESIGN_MEM.get("cpu", BENCH["cpu"]["max_rss_gb"]) + 0.2
m_gpu = 1.25 * DESIGN_MEM.get("cuda", 2 * m_cpu) + 0.2
vis = os.environ.get("CUDA_VISIBLE_DEVICES")
VISIBLE = [d.strip() for d in vis.split(",") if d.strip()] if vis is not None else [str(i) for i in range(len(GPUS))]
N_GPUS = len(VISIBLE) if "cuda" in BACKENDS else 0           # GPUs that the workers may use
if N_GPUS > 1:                                                # GPU workers are spread over the visible GPUs
    os.environ["TRIMFIELDS_GPU_DEVICES"] = ",".join(VISIBLE)
gpu_mem = min((float(g.split(",")[1].split()[0]) / 1024 for g in GPUS), default=0.0)   # GB per GPU
max_gpu_workers = N_GPUS * int(gpu_mem / 0.8)                 # about 0.8 GB of GPU memory per worker


def total_workers(n_g):
    if WORKERS != "auto":
        return int(WORKERS)
    return max(1, min(CPUS - 1, n_g + int(max(0.0, MEM - n_g * m_gpu) / m_cpu)))


def throughput(n_g):
    n = total_workers(n_g)
    gpu = min(n_g / (t_rest + BENCH["cuda"]["t_indicator"]), N_GPUS / BENCH["cuda"]["t_indicator"]) if n_g else 0.0
    return gpu + (n - n_g) / (t_rest + t_cpu)


if N_GPUS:
    if GPU_WORKERS == "auto":
        cands = [n for n in range(0, min(CPUS - 1, max_gpu_workers) + 1) if total_workers(n) >= n
                 and n * m_gpu <= max(MEM, m_gpu)]
        N_GPU = max(cands, key=lambda n: (round(throughput(n), 6), -n))
    else:
        N_GPU = int(GPU_WORKERS)
    N_TOTAL = max(total_workers(N_GPU), N_GPU)
    print(f"refinement steps per second: all workers on the CPU {throughput(0):.1f}, chosen split "
          f"{throughput(N_GPU):.1f}")
else:
    if GPU_WORKERS not in ("auto", "0"):
        print("GPU workers requested, but JAX cannot use the GPU: all workers run on the CPU")
    N_GPU, N_TOTAL = 0, total_workers(0)
print(f"workers: {N_TOTAL} ({N_GPU} with the JAX part on the GPU); memory per worker {m_cpu:.1f} GB (CPU), "
      f"{m_gpu:.1f} GB (GPU); {MEM:.0f} GB available")


def rate(kind):                                               # designs per second
    r = 0.0
    if ("cuda", kind) in DESIGN_T:
        r += N_GPU / DESIGN_T[("cuda", kind)]
    if ("cpu", kind) in DESIGN_T:
        r += (N_TOTAL - N_GPU) / DESIGN_T[("cpu", kind)]
    return r


if MODE == "full" and rate("planar") > 0 and rate("shell") > 0:
    eta_h = (N_PLANAR / rate("planar") + N_SHELL / rate("shell")) / 3600 * 1.15    # 15 % for tail and variation
    print(f"estimated time for the full dataset: {eta_h:.0f} h ({eta_h / 24:.1f} days); designs differ in "
          "size, so treat this as a rough figure (the progress bar gives the running estimate)")
elif MODE == "smoke":
    print("smoke mode: timings use small meshes, so no run-time estimate for the full dataset is given")
""")

md("""
## 7. Smoke run

One design per family with the production settings (smaller meshes in `smoke` mode), consolidated like the full
dataset. It checks the worker pool, the GPU assignment, STEP export and consolidation on this machine.
""")
code(r"""
smoke = P.smoke_config(cfg)
if MODE == "smoke":
    smoke["n_el_planar"], smoke["n_el_shell"] = 900, 900
t0 = time.monotonic()
res = P.run_all(smoke, P.task_list(smoke), N_TOTAL, n_gpu_workers=N_GPU)
bad = [r for r in res if r["status"] != "ok"]
print(f"smoke run: {len(res)} designs in {time.monotonic() - t0:.0f} s, {len(bad)} not ok")
for r in bad:
    print("  ", r)
assert not bad, "the smoke run failed; see the messages above and <out>/_smoke/failures"
SMOKE_SUMMARY = P.consolidate(smoke, min(N_TOTAL, 8))
print(json.dumps(SMOKE_SUMMARY, indent=1))
""")

md("""
## 8. Full generation (resumable)

All designs are generated by the worker pool, longest (shell) designs first. Designs whose problems keep failing
are retried in later rounds with new seeds (the seed cursor is kept in `<out>/rounds.json`, so a rerun continues
with unused seeds). Progress is also written to `<out>/progress.log`; a design that exceeds `task_timeout` is
stopped by its watchdog and retried in the next round. The notebook stops here, before consolidation and
packaging, if designs are still missing; running it again resumes the generation.
""")
code(r"""
if MODE == "full":
    t0 = time.monotonic()
    ROUNDS = P.generate_until_complete(cfg, N_TOTAL, rounds=10, n_gpu_workers=N_GPU)
    missing = [t for t in TASKS if not P.is_done(cfg, t)]
    print(f"generation finished in {(time.monotonic() - t0) / 3600:.1f} h; missing designs: {len(missing)}")
    if missing:
        raise RuntimeError(f"{len(missing)} designs are still missing, e.g. {missing[:5]}. Run the notebook again: "
                           f"it resumes with new seeds for them. Reasons are in {OUT_DIR}/failures.")
else:
    ROUNDS = []
    print("smoke mode: the full generation is skipped")
""")

md("## 9. Consolidation")
code(r"""
TARGET = cfg if MODE == "full" else smoke
t0 = time.monotonic()
SUMMARY = P.consolidate(TARGET, N_TOTAL)
print(f"consolidated in {time.monotonic() - t0:.0f} s")
print(json.dumps(SUMMARY, indent=1))
""")

md("## 10. Quality report and gallery")
code(r"""
import pandas as pd

cdir = os.path.join(TARGET["out_dir"], "consolidated")
frames = [pd.read_csv(os.path.join(cdir, f"{g}_index.csv")) for g in ("planar", "shell")
          if os.path.exists(os.path.join(cdir, f"{g}_index.csv"))]
IDX = pd.concat(frames, ignore_index=True)
IDX["J_final_over_thr"] = IDX["J_final"] / IDX["J_thr"]            # NaN for random designs
IDX["J_final_over_simp"] = IDX["J_final"] / IDX["J_simp"]
IDX["J_f2_over_fine"] = IDX["J_final"] / IDX["J_fine"]              # production against finer evaluation (test)
QUALITY = IDX.groupby(["kind", "family"]).agg(
    designs=("row", "size"), resampled=("attempt", lambda a: float((a > 0).mean())),
    refined_final=("final_source", lambda s: float((s == "refine").mean())),
    J_final_over_thr=("J_final_over_thr", "median"), J_final_over_simp=("J_final_over_simp", "median"),
    J_f2_over_fine=("J_f2_over_fine", "median"), vm_max_loops=("vm_max_loops", "median"),
    loops=("loops_final", "median"), V=("V_final", "median"), valid=("valid", "mean"),
    step_valid=("step_valid", "mean"), t_median_s=("t_total", "median"))
QUALITY.to_csv(os.path.join(cdir, "quality.csv"))
pd.set_option("display.precision", 3)
display(QUALITY)
n_fail = len(os.listdir(os.path.join(TARGET["out_dir"], "failures"))) \
    if os.path.isdir(os.path.join(TARGET["out_dir"], "failures")) else 0
print(f"{len(IDX)} designs, {n_fail} failure records, all valid: {bool(IDX['valid'].all())}, "
      f"all STEP valid: {bool(IDX['step_valid'].all())}")
""")
code(r"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath

from trimfields.hosts import Host
from trimfields.loops import sample_loop


def material(prob, loops, n):
    host = Host.from_dict(prob["host"]).build(8, 8)
    c = np.linspace(0, 1, n)
    U, V = np.meshgrid(c, c, indexing="ij")
    uv = np.stack([U.ravel(), V.ravel()], 1)
    X = host.eval(uv[:, 0], uv[:, 1])
    mat = np.ones(len(uv), bool)
    for C in loops:
        mat &= ~MPath(sample_loop(np.asarray(C), 20)).contains_points(uv)
    for ps in prob["passive"]:
        if ps["kind"] == "circle":
            mat &= np.hypot(X[:, 0] - ps["centre"][0], X[:, 1] - ps["centre"][1]) >= ps["radius"]
        else:
            mat &= ~MPath(np.asarray(ps["uv"])).contains_points(uv)
    return host, X.reshape(n, n, -1), mat.reshape(n, n)


def draw(ax, path, n=120):
    d = np.load(path)
    prob, loops = json.loads(str(d["problem"])), d["loops_final"]
    host, X, mat = material(prob, loops, n)
    face = np.where(mat, 1.0, np.nan)
    if X.shape[2] == 2:
        ax.pcolormesh(X[..., 0], X[..., 1], face[:-1, :-1], cmap="Greys", vmin=0, vmax=1.4, shading="flat")
        for C in loops:
            s = sample_loop(np.asarray(C), 30)
            s = s[np.all((s >= 0) & (s <= 1), 1)]
            if len(s):
                P_ = host.eval(s[:, 0], s[:, 1])
                ax.plot(P_[:, 0], P_[:, 1], ".", ms=0.6, color="#c0392b")
        ax.set_aspect("equal")
        ax.axis("off")
    else:
        fc = np.zeros(mat.shape + (4,))
        fc[mat] = (0.35, 0.45, 0.6, 1.0)
        ax.plot_surface(X[..., 0], X[..., 1], X[..., 2], facecolors=fc, rstride=1, cstride=1, linewidth=0,
                        antialiased=False, shade=False)
        ax.set_box_aspect(np.ptp(X.reshape(-1, 3), 0) + 1e-9)
        ax.axis("off")
    ax.set_title(f"{prob['split']} {prob['family']} {os.path.basename(path)[:-4]}", fontsize=11)


rng = np.random.default_rng(0)
rows = []
for kind in ("planar", "shell"):
    sub = IDX[IDX["path"].str.contains("shell") == (kind == "shell")]      # shell, shell_ood, rand_shell, rand_shell_ood
    fams = sorted(sub["family"].unique())
    n = 12 if kind == "planar" else 6
    picks, used = [], set()
    for i in range(n):                                         # cycle through the families, no repeats
        pool = sub[(sub["family"] == fams[i % len(fams)]) & ~sub["row"].isin(used)] if fams else sub.iloc[:0]
        if len(pool):
            row = pool.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            used.add(row["row"])
            picks.append(row)
    rows += [("planar", picks[:6]), ("planar", picks[6:])] if kind == "planar" else [("shell", picks)]
fig = plt.figure(figsize=(18, 9))
for r, (kind, picks) in enumerate(rows):
    for c, row in enumerate(picks):
        ax = fig.add_subplot(3, 6, 6 * r + c + 1, projection="3d" if kind == "shell" else None)
        draw(ax, os.path.join(TARGET["out_dir"], row["path"]))
fig.tight_layout()
GALLERY = os.path.join(cdir, "gallery.png")
fig.savefig(GALLERY, dpi=110)
plt.close(fig)
from IPython.display import Image, display

display(Image(GALLERY))
""")

md("## 11. Manifest and archives")
code(r"""
import hashlib
import tarfile


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


root = pathlib.Path(TARGET["out_dir"])
MANIFEST = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": MODE, "trimfields_version": trimfields.__version__,
            "config": TARGET, "machine": MACHINE, "benchmark": BENCH, "design_times": {f"{k[0]}/{k[1]}": v
                                                                                       for k, v in DESIGN_T.items()},
            "design_memory_gb": DESIGN_MEM,
            "workers": {"total": N_TOTAL, "gpu": N_GPU}, "checks": CHECKS, "rounds": ROUNDS, "summary": SUMMARY,
            "quality": json.loads(QUALITY.to_json(orient="index")),
            "sources": {f"trimfields/{m}.py": sha256(WORK / "trimfields" / f"{m}.py") for m in
                        [p.stem for p in sorted((WORK / "trimfields").glob("*.py"))]},
            "notebook_hours": (time.monotonic() - T_START) / 3600}
with open(root / "manifest.json", "w") as f:
    json.dump(MANIFEST, f, indent=1, default=str)
shutil.copytree(WORK / "trimfields", root / "trimfields_source", dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__"))

arch = root.parent / (root.name + "_archives")
arch.mkdir(exist_ok=True)
PARTS = {"consolidated": ["consolidated", "manifest.json", "progress.log", "failures", "trimfields_source"],
         "designs": ["designs"], "step": ["step"]}
sums = []
for name, items in PARTS.items():
    out = arch / f"trimfields_{name}.tar.gz"
    t0 = time.monotonic()
    with tarfile.open(out, "w:gz", compresslevel=6) as tar:
        for it in items:
            if (root / it).exists():
                tar.add(root / it, arcname=f"{root.name}/{it}")
    sums.append(f"{sha256(out)}  {out.name}")
    print(f"{out.name}: {out.stat().st_size / 2**20:,.1f} MB in {time.monotonic() - t0:.0f} s")
(arch / "SHA256SUMS").write_text("\n".join(sums) + "\n")
print(f"archives in {arch}; total notebook time {(time.monotonic() - T_START) / 3600:.2f} h")
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbf.validate(nb)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
