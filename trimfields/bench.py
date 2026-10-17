"""Timings used to choose the JAX backend and the worker count, and to estimate the total run time.

Run as a module in a fresh process so that JAX_PLATFORMS and the thread settings of a worker apply:
    python -m trimfields.bench indicator <n_el> <reps>
    python -m trimfields.bench design <split> <family> <idx> <out_dir> [small]
The last line of the output is 'RESULT <json>'.
"""
import copy
import json
import resource
import sys
import time

import numpy as np


def indicator_timing(n_el=2700, reps=10):
    """Time of the JAX part of one refinement step (indicator of a new design state, its gradient, volume and volume
    gradient) and of one sparse solve (the CPU part), on a production-size planar model with eight loops."""
    import jax

    from . import indicator as I
    from .checks import _cantilever, _circle_loops
    from .mechanics import USE_PARDISO, Model
    from .pipeline import CONFIG
    from .refine import Refiner
    prob, host, apply_bc = _cantilever(n_el)
    m4 = apply_bc(prob, Model.on_grid(host, 4))
    ind = I.Indicator(m4, prob, 16, 16)
    ref = Refiner(m4, ind, CONFIG["refine"])
    centres = [(0.15 + 0.23 * i, 0.3 + 0.4 * j) for i in range(4) for j in range(2)]
    Cz, act = ref._slots(_circle_loops(centres, [0.07] * 8))
    ct = np.ones(ind.m_pad)

    def step(k):                                  # one refinement step: new state, gradient, volume and gradient
        C = Cz + 1e-9 * k
        st = ind.state(C, act)
        ind.chi_vjp(st, ct)
        ind.volume(C, act)
        return st["chi"]

    t = time.perf_counter()
    step(0)
    t_compile = time.perf_counter() - t
    t = time.perf_counter()
    for k in range(reps):
        E = step(k + 1)
    t_ind = (time.perf_counter() - t) / reps
    E = I.EMIN + np.asarray(E)[: ind.m] * (1 - I.EMIN)
    m4.solve(E)                                   # warm-up (loads MKL)
    t = time.perf_counter()
    for k in range(3):
        m4.solve(E * (1 + 1e-3 * k))
    t_solve = (time.perf_counter() - t) / 3
    return {"backend": jax.default_backend(), "device": str(jax.devices()[0]), "points": int(ind.m),
            "t_indicator": t_ind, "t_solve": t_solve, "t_compile": t_compile, "pardiso": bool(USE_PARDISO),
            "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}


def design_timing(split, family, idx, out_dir, small=False):
    """Wall time of one complete design (all stages, including rejected attempts)."""
    from .pipeline import CONFIG, design_path, generate_one, smoke_config
    cfg = copy.deepcopy(CONFIG)
    cfg["out_dir"] = out_dir
    if small:                                     # smoke-mode timing: short iterations and small meshes
        cfg = smoke_config(cfg)
        cfg["out_dir"] = out_dir
        cfg["n_el_planar"], cfg["n_el_shell"] = 900, 900
    t = time.perf_counter()
    r = generate_one((split, family, idx), cfg)
    r["wall"] = time.perf_counter() - t
    r["max_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    if r["status"] == "ok":
        m = json.loads(str(np.load(design_path(cfg, split, family, idx))["metrics"]))
        r.update({"attempt": m["attempt"], "times": m["times"], "nel": m["nel"], "ndof": m["ndof"],
                  "J_final_over_J_thr": m["J_final"] / m["J_thr"] if m.get("J_thr") else None})
    return r


if __name__ == "__main__":
    if sys.argv[1] == "indicator":
        out = indicator_timing(int(sys.argv[2]), int(sys.argv[3]))
    else:
        out = design_timing(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5],
                            small=len(sys.argv) > 6 and sys.argv[6] == "small")
    print("RESULT " + json.dumps(out, default=str), flush=True)
