"""Dataset generation: one design end to end, a resumable process pool, and consolidation into arrays.

Two kinds of designs are generated. Optimized designs follow the TrimTO pipeline (isogeometric SIMP, loop fit,
CAD-native refinement, admissibility test, STEP export). Random designs (the `rand` splits) are admissible loop
sets without optimization, analysed and exported in the same way; loop sets that cut the load path are rejected.
Every design stores the reference fields of fields.py and the compliance of the untrimmed host under its loads
(J_host); designs of the test splits also store the compliance of a finer evaluation (ref_factor for planar faces,
ref_factor_shell for shells).
"""
import copy
import hashlib
import json
import os
import time
import traceback

import numpy as np

CONFIG = {
    "out_dir": "trimfields_data",
    "seed": 20260927,
    # designs per split and family: 14,300 optimized (as the TrimTO pipeline, under continuum boundary conditions)
    # and 11,000 random designs; 25,300 in total (about 280 core-hours at the costs measured on the 570-design sample)
    "counts": {
        "id": {"rect": 2000, "sector": 2000, "plate": 2000, "arc": 2000, "taper": 2000, "bolted": 2000},
        "ood_load": {"rect": 100, "sector": 100, "plate": 100, "arc": 100, "taper": 100},
        "ood_host": {"freeform": 250, "ell": 250},
        "shell": {"cyl": 400, "saddle": 300, "dome": 300, "torus": 200},
        "shell_ood": {"hood": 100},
        "rand": {"rect": 1500, "sector": 1500, "plate": 1500, "arc": 1500, "taper": 1500, "bolted": 1500},
        "rand_host": {"freeform": 500, "ell": 500},
        "rand_shell": {"cyl": 200, "saddle": 200, "dome": 200, "torus": 200},
        "rand_shell_ood": {"hood": 200},
    },
    "test_every": 10,                       # every 10th in-distribution design goes to the test split
    "n_el_planar": 2700,
    "n_el_shell": 2304,
    "thickness_rel": 0.05,                  # shell thickness / sqrt(area)
    "simp": {"penal": 3.0, "maxit": 200, "tol": 0.01, "move": 0.2, "rmin_h": 2.0},
    "simp_shell": {"penal": 3.0, "maxit": 250, "tol": 0.005, "move": 0.2, "rmin_h": 1.5, "beta_max": 16.0},
    "n_cp": 16,
    "k_max": 16,
    "refine": {"maxit": 200, "n_top": 25, "lr0": 0.25, "lr1": 0.01, "alpha": 2e-3, "beta": 0.1, "gamma": 1.0,
               "mu": 50.0, "w_min_h": 1.5, "nucleation": True, "nucleation_tau": 0.02,
               "nucleation_radius_h": 1.5, "min_iters": 80, "stop_rel": 2e-4, "gamma_keep": 1.0,
               "move_accept_tol": 0.0},
    # random designs: 1 to max_loops loops, clearance w_min_h elements, volume fraction at least min_volume, and a
    # compliance of at most max_compliance_host times that of the untrimmed host (an intact load path)
    "random": {"max_loops": 8, "w_min_h": 1.5, "min_volume": 0.15, "max_compliance_host": 50.0},
    "eval_factor": 2,                       # production evaluation: f times as many elements per direction
    "ref_factor": 4,                        # finer evaluation stored for the planar designs of the test splits
    "ref_factor_shell": 3,                  # the same for shells (their f = 4 evaluation needs 2.6 GB per worker)
    "loop_samples": 64,                     # boundary samples per loop (fields.loop_samples)
    "export_step": True,
    "bucket": 8192,
    "vol_tol": 5e-3,                        # admissible designs: |V - V*| below this (crisp volume fraction)
    "max_attempts": 5,                      # problems without an admissible trimmed design are resampled
    "attempt_offset": 0,                    # raised by retry rounds so that they draw new problems
    "task_timeout": 3600,                   # seconds per design (all attempts); the worker is killed beyond it
}

RANDOM_SPLITS = ("rand", "rand_host", "rand_shell", "rand_shell_ood")
PLANAR_SPLITS = ("id", "ood_load", "ood_host", "rand", "rand_host")
SHELL_SPLITS = ("shell", "shell_ood", "rand_shell", "rand_shell_ood")
TRAIN_TEST_SPLITS = ("id", "shell", "rand", "rand_shell")   # every test_every-th design is a test design


class Rejected(RuntimeError):
    """The sampled problem gave no admissible trimmed design; the next attempt resamples the problem."""


def design_path(cfg, split, family, idx, ext="npz"):
    sub = "designs" if ext == "npz" else "step"
    return os.path.join(cfg["out_dir"], sub, split, family, f"{idx:05d}.{ext}")


def task_list(cfg):
    tasks = []
    for split, fam_counts in cfg["counts"].items():
        for family, n in fam_counts.items():
            tasks += [(split, family, i) for i in range(n)]
    return tasks


def sub_split(cfg, split, idx):
    """Name of the split in the consolidated index: train/test parts of the in-distribution splits."""
    if split in TRAIN_TEST_SPLITS:
        return f"{split}_test" if idx % cfg["test_every"] == 0 else f"{split}_train"
    return split


def is_test(cfg, split, idx):
    return not sub_split(cfg, split, idx).endswith("_train")


def _seed(cfg, split, family, idx, attempt):
    h = hashlib.sha256(f"{cfg['seed']}/{split}/{family}/{idx}/{attempt}".encode()).digest()
    return int.from_bytes(h[:8], "little")


# ------------------------------------------------------------------------------------------------ one design
def admissible(loops, r, vstar, keep_geoms, vol_tol, j_max=None):
    """'ok' or the reason a loop set is not an admissible trimmed design. j_max (twice the compliance of the
    thresholded density design) rejects designs whose load path is broken."""
    from shapely.geometry import Polygon

    from .loops import loop_collisions, sample_loop, self_intersections
    from .problems import covers_keep
    if not (np.isfinite(r["J"]) and r["J"] > 0):
        return f"invalid compliance {r['J']}"
    if j_max is not None:
        if not (np.isfinite(j_max) and j_max > 0):
            return f"invalid reference compliance for the load-path test ({j_max})"
        if r["J"] > j_max:
            return f"compliance {r['J']:.4g} above {j_max:.4g} (broken load path)"
    if not abs(r["V"] - vstar) <= vol_tol:                  # also rejects NaN
        return f"volume {r['V']:.4f} vs {vstar:.4f}"
    if any(self_intersections(C) for C in loops):
        return "self-intersection"
    if len(loops) > 1 and loop_collisions(loops):
        return "loop collision"
    if any(covers_keep(Polygon(sample_loop(np.asarray(C), 20)).buffer(0), keep_geoms) for C in loops):
        return "loop covers a support or load"
    return "ok"


def _strip_passive(raw, prob, nel):
    """Contours minus the passive regions: an L-shape cut-out, or hole-plus-ring discs grown by one element.
    Pieces below two element areas are dropped (contours of a passive hole alone)."""
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union

    from .problems import passive_discs_uv
    el = 1.0 / (nel[0] * nel[1])
    pas = [Polygon(ps["uv"]) for ps in prob["passive"] if ps["kind"] == "polygon"]
    pas += [Polygon(d).buffer(1.0 / max(nel)) for d in passive_discs_uv(prob)]
    pas_u = unary_union(pas)
    out = []
    for a in raw:
        g = Polygon(a).buffer(0).difference(pas_u)
        for piece in (g.geoms if isinstance(g, MultiPolygon) else [g]):
            if piece.is_empty or piece.area <= 2 * el:
                continue
            if any(ps["kind"] == "circle" for ps in prob["passive"]):
                out.append(np.asarray(piece.exterior.coords)[:-1])
            else:                                   # L-shape: keep the original contour (union of voids)
                out.append(np.asarray(a))
                break
    return out


def is_done(cfg, task):
    """A design is complete when its npz exists and, with STEP export, its STEP file too."""
    return os.path.exists(design_path(cfg, *task)) and (
        not cfg["export_step"] or os.path.exists(design_path(cfg, *task, ext="step")))


def fit_at_volume(host, rho, level, prob, nel, cfg, ev, iters=10):
    """Loops fitted to the density contours of a level. If the fitted loops miss V* by more than the tolerance
    (for instance where passive regions are cut out of the contours), the level is adjusted by bisection until the
    crisp volume of the fitted loops matches V*. Returns (loops, level)."""
    from .loops import extract_void_loops, fit_loops, in_square_area

    def fit(lev):
        raw = extract_void_loops(host, rho, level=lev)
        if prob["passive"]:
            raw = _strip_passive(raw, prob, nel)
        loops = sorted(fit_loops(host, raw, cfg["n_cp"]), key=in_square_area, reverse=True)[: cfg["k_max"]]
        return loops, ev.volume(ev.loop_material(loops))

    loops, V = fit(level)
    tol = 0.4 * cfg["vol_tol"]
    if abs(V - prob["vstar"]) <= tol:
        return loops, level
    best = (abs(V - prob["vstar"]), loops, level)
    lo, hi = (level, min(0.95, level + 0.4)) if V > prob["vstar"] else (max(0.05, level - 0.4), level)
    for _ in range(iters):                                # more material than V*: raise the level, and vice versa
        mid = 0.5 * (lo + hi)
        L_, V_ = fit(mid)
        if abs(V_ - prob["vstar"]) < best[0]:
            best = (abs(V_ - prob["vstar"]), L_, mid)
        if abs(V_ - prob["vstar"]) <= tol:
            break
        lo, hi = (mid, hi) if V_ > prob["vstar"] else (lo, mid)
    return best[1], best[2]


def _setup(prob, cfg):
    """Host, mesh and thickness of a sampled problem."""
    from .hosts import SHELL, Host
    shell = prob["family"] in SHELL
    host = Host.from_dict(prob["host"])
    nel = host.mesh_for(cfg["n_el_shell"] if shell else cfg["n_el_planar"])
    host.build(*nel)
    nel = (host.nelu, host.nelv)
    thick = cfg["thickness_rel"] * np.sqrt(host.area) if shell else None
    prob["thickness"] = thick
    return host, nel, thick, shell


def _optimized_design(prob, host, nel, thick, shell, cfg, T):
    """SIMP, thresholded evaluation, loop fit and CAD-native refinement; returns the final design and its records."""
    from .evaluate import Evaluator
    from .indicator import Indicator
    from .mechanics import Model
    from .problems import apply_bc, keep_cps, keep_geometry, keep_samples, passive_masks_cp
    from .refine import Refiner
    from .simp import run_simp, run_simp_projected

    # 1. isogeometric SIMP
    t = time.perf_counter()
    m3 = apply_bc(prob, Model.on_grid(host, 3, thick))
    void, solid = passive_masks_cp(prob, host) if prob["passive"] else (None, np.zeros(host.ncu * host.ncv, bool))
    solid = solid | keep_cps(prob, host)    # non-design solid under supports and loads
    keep_g = keep_geometry(prob)
    h_el = np.sqrt(host.area / (nel[0] * nel[1]))
    if shell:
        sc = cfg["simp_shell"]
        rho, sinfo = run_simp_projected(m3, prob["vstar"], sc["rmin_h"] * h_el, sc["penal"], sc["maxit"],
                                        sc["tol"], sc["move"], beta_max=sc["beta_max"], solid=solid)
    else:
        sc = cfg["simp"]
        rho, sinfo = run_simp(m3, prob["vstar"], sc["rmin_h"] * h_el, void, solid, sc["penal"], sc["maxit"],
                              sc["tol"], sc["move"])
    if not np.isfinite(sinfo["J_simp"]) or sinfo["J_simp"] <= 0:
        raise RuntimeError("SIMP failed")
    del m3
    T["simp"] = time.perf_counter() - t

    # 2. crisp evaluation of the thresholded density
    t = time.perf_counter()
    ev = Evaluator(prob, nel, thick, cfg["eval_factor"])
    th = ev.threshold_for_volume(host, rho, prob["vstar"])
    r_thr = ev.evaluate(ev.density_material(rho, th))
    T["eval_thr"] = time.perf_counter() - t
    if not abs(r_thr["V"] - prob["vstar"]) <= 4 * cfg["vol_tol"]:      # the target volume is unattainable by a level
        raise Rejected(f"thresholded density volume {r_thr['V']:.4f} vs {prob['vstar']:.4f}")
    J_host = _host_compliance(ev, T)

    # 3. loops fitted to the density contour
    t = time.perf_counter()
    loops_fit, level_fit = fit_at_volume(host, rho, th, prob, nel, cfg, ev)
    r_fit = ev.evaluate(ev.loop_material(loops_fit))
    T["fit"] = time.perf_counter() - t

    # 4. CAD-native refinement
    t = time.perf_counter()
    m4 = apply_bc(prob, Model.on_grid(host, 4, thick))
    ind = Indicator(m4, prob, cfg["n_cp"], cfg["k_max"], bucket=cfg["bucket"],
                    keep_uv=keep_samples(keep_g, 0.5 / max(nel)))
    ref = Refiner(m4, ind, cfg["refine"], keep_g)
    loops_ref, rinfo = ref.run(loops_fit, prob["vstar"], cfg["refine"]["maxit"])
    del ref, ind, m4
    T["refine"] = time.perf_counter() - t
    t = time.perf_counter()
    r_ref = ev.evaluate(ev.loop_material(loops_ref), keep_model=True)
    T["eval_ref"] = time.perf_counter() - t
    # final design: the admissible one of refined and fitted loops with the lower crisp compliance
    adm = {name: admissible(L, r, prob["vstar"], keep_g, cfg["vol_tol"], 2.0 * r_thr["J"])
           for name, L, r in (("refine", loops_ref, r_ref), ("fit", loops_fit, r_fit))}
    cands = [(r["J"], name, L, r) for name, L, r in (("refine", loops_ref, r_ref), ("fit", loops_fit, r_fit))
             if adm[name] == "ok"]
    if not cands:
        raise Rejected(f"no admissible design (refine: {adm['refine']}, fit: {adm['fit']})")
    _, final_source, loops_fin, r_fin = min(cands, key=lambda c_: c_[0])
    if final_source == "fit":                                 # one evaluation model at a time (memory per worker)
        r_ref.pop("model", None)
        r_ref.pop("U", None)
        t = time.perf_counter()
        r_fin = ev.evaluate(ev.loop_material(loops_fit), keep_model=True)
        T["eval_ref"] += time.perf_counter() - t
    metrics = {"kind": "optimized", "simp": {k: v for k, v in sinfo.items()}, "threshold": th, "J_host": J_host,
               "fit_level": level_fit, "J_thr": r_thr["J"], "V_thr": r_thr["V"], "J_fit": r_fit["J"],
               "V_fit": r_fit["V"], "J_ref": r_ref["J"], "V_ref": r_ref["V"],
               "cut_elements_ref": r_ref["cut_elements"], "final_source": final_source, "admissible": adm,
               "loops_fit": len(loops_fit), "loops_ref": len(loops_ref),
               "refine": {k: v for k, v in rinfo.items() if k != "history"}}
    arrays = {"rho": rho.astype(np.float32), "loops_fit": loops_fit, "loops_ref": loops_ref,
              "history": rinfo["history"].astype(np.float32)}
    return loops_fin, r_fin, ev, keep_g, metrics, arrays


def _host_compliance(ev, T):
    """Crisp compliance of the untrimmed host (passive holes excluded) under the problem's loads: the scale of the
    compliance of every design on that problem, and the reference of the load-path test of random designs."""
    t = time.perf_counter()
    J = ev.evaluate(ev.loop_material([]))["J"]
    T["eval_host"] = time.perf_counter() - t
    if not (np.isfinite(J) and J > 0):
        raise Rejected(f"invalid host compliance {J}")
    return J


def _random_design(prob, host, nel, thick, cfg, rng, T):
    """Random admissible loop set, analysed with the crisp evaluator; V* is set to its volume fraction. Loop sets
    that cut the load path (compliance above max_compliance_host times that of the untrimmed host) are rejected."""
    from .evaluate import Evaluator
    from .problems import keep_geometry, random_loops
    t = time.perf_counter()
    rc = cfg["random"]
    keep_g = keep_geometry(prob)
    loops = random_loops(prob, host, nel, rng, cfg["n_cp"], rc["max_loops"], rc["w_min_h"], keep_g)
    if not loops:
        raise Rejected("no admissible random loop")
    T["random"] = time.perf_counter() - t
    ev = Evaluator(prob, nel, thick, cfg["eval_factor"])
    J_host = _host_compliance(ev, T)
    t = time.perf_counter()
    r = ev.evaluate(ev.loop_material(loops), keep_model=True)
    T["eval_ref"] = time.perf_counter() - t
    if not (np.isfinite(r["V"]) and r["V"] >= rc["min_volume"]):
        raise Rejected(f"random design too light: volume fraction {r['V']:.3f}")
    prob["vstar"] = float(r["V"])
    adm = admissible(loops, r, prob["vstar"], keep_g, cfg["vol_tol"], rc["max_compliance_host"] * J_host)
    if adm != "ok":
        raise Rejected(f"random design not admissible: {adm}")
    metrics = {"kind": "random", "final_source": "random", "admissible": {"random": adm}, "J_host": J_host,
               "cut_elements_ref": r["cut_elements"], "loops_fit": 0, "loops_ref": 0}
    arrays = {"rho": np.zeros(0, np.float32), "loops_fit": [], "loops_ref": [], "history": np.zeros((0, 4), np.float32)}
    return loops, r, ev, keep_g, metrics, arrays


def _reference_fields(prob, host, nel, thick, loops, r_fin, ev, cfg, test, T):
    """Reference fields of the final design: evaluation-mesh displacement, boundary samples along the loops, the
    design fields, and the finer evaluation on the test splits."""
    from .evaluate import Evaluator
    from .fields import design_field, loop_samples
    t = time.perf_counter()
    model, U = r_fin["model"], r_fin["U"]
    ls = loop_samples(model, U, loops, cfg["loop_samples"], prob)
    arrays = {"u_eval": U.astype(np.float32), "design_field": design_field(prob, loops).astype(np.float16)}
    arrays.update({f"loop_{k}": v for k, v in ls.items()})
    metrics = {"eval_factor": cfg["eval_factor"], "eval_nel": [ev.host.nelu, ev.host.nelv],
               "eval_ndof": int(model.ndof),
               "vm_max_loops": float(ls["vm"][ls["inside"]].max()) if ls["inside"].any() else None}
    T["fields"] = time.perf_counter() - t
    if test:
        t = time.perf_counter()
        rf = cfg["ref_factor_shell"] if thick is not None else cfg["ref_factor"]
        ev_fine = Evaluator(prob, nel, thick, rf)
        r_fine = ev_fine.evaluate(ev_fine.loop_material(loops))
        metrics.update({"ref_factor": rf, "J_fine": r_fine["J"], "V_fine": r_fine["V"]})
        T["eval_fine"] = time.perf_counter() - t
    else:
        metrics.update({"ref_factor": None, "J_fine": None, "V_fine": None})
    return arrays, metrics


def generate_one(task, cfg):
    split, family, idx = task
    out = design_path(cfg, split, family, idx)
    if is_done(cfg, task):
        return {"task": task, "status": "exists"}
    from .loops import canonical, loop_collisions, self_intersections
    from .problems import sample_problem

    random_design = split in RANDOM_SPLITS
    last_err, rejections = None, []
    for attempt in range(cfg["max_attempts"]):
        rng = np.random.default_rng(_seed(cfg, split, family, idx, attempt + cfg.get("attempt_offset", 0)))
        T = {}
        try:
            t = time.perf_counter()
            prob = sample_problem(family, rng, "ood_load" if split == "ood_load" else "id")
            prob["split"] = split
            host, nel, thick, shell = _setup(prob, cfg)
            T["setup"] = time.perf_counter() - t
            if random_design:
                loops_fin, r_fin, ev, keep_g, met, arrays = _random_design(prob, host, nel, thick, cfg, rng, T)
            else:
                loops_fin, r_fin, ev, keep_g, met, arrays = _optimized_design(prob, host, nel, thick, shell, cfg, T)

            # validity, reference fields and export
            metrics = {
                "split": split, "family": family, "idx": idx, "attempt": attempt + cfg.get("attempt_offset", 0),
                "test": is_test(cfg, split, idx), "vstar": prob["vstar"],
                "nel": list(nel), "ndof": int(host.ncu * host.ncv * (3 if shell else 2)), "area": host.area,
                "thickness": thick, "J_final": r_fin["J"], "V_final": r_fin["V"], "rejections": rejections,
                "loops_final": len(loops_fin),
                "self_intersections_final": int(sum(self_intersections(C) for C in loops_fin)),
                "collisions_final": int(loop_collisions(loops_fin)) if loops_fin else 0,
            }
            metrics.update(met)
            metrics["valid"] = metrics["self_intersections_final"] == 0 and metrics["collisions_final"] == 0
            f_arrays, f_metrics = _reference_fields(prob, host, nel, thick, loops_fin, r_fin, ev, cfg,
                                                    metrics["test"], T)
            metrics.update(f_metrics)
            r_fin.pop("model", None)
            if cfg["export_step"]:
                t = time.perf_counter()
                from .export_step import HAVE_OCP, export_design
                if not HAVE_OCP:
                    raise RuntimeError("OCP (cadquery-ocp) is not importable; install it or set export_step=False")
                spath = design_path(cfg, split, family, idx, "step")
                stmp = f"{spath[:-5]}.{os.getpid()}.tmp.step"   # one file per writer, renamed when complete
                try:
                    st = export_design(prob, loops_fin, stmp)
                except Exception as ex:  # noqa: BLE001
                    if os.path.exists(stmp):
                        os.remove(stmp)
                    raise Rejected(f"STEP export failed: {type(ex).__name__}: {ex}") from ex
                st["area_rel_err"] = abs(st["area"] - r_fin["V"] * host.area) / (r_fin["V"] * host.area)
                if not st["valid"] or not st["area_rel_err"] <= 0.02:
                    os.remove(stmp)
                    raise Rejected(f"STEP check failed: valid {st['valid']}, area error {st['area_rel_err']:.3g}")
                metrics["step"] = st
                T["step"] = time.perf_counter() - t
            metrics["times"] = T
            metrics["t_total"] = float(sum(T.values()))
            X = canonical(loops_fin, cfg["k_max"])
            pack = lambda Ls: (np.stack(Ls) if len(Ls) else np.zeros((0, cfg["n_cp"], 2))).astype(np.float64)  # noqa: E731
            os.makedirs(os.path.dirname(out), exist_ok=True)
            tmp = f"{out[:-4]}.{os.getpid()}.tmp.npz"
            np.savez_compressed(tmp, problem=json.dumps(prob), metrics=json.dumps(metrics, default=float),
                                rho=arrays["rho"], loops_fit=pack(arrays["loops_fit"]),
                                loops_ref=pack(arrays["loops_ref"]), loops_final=pack(loops_fin),
                                x=X.astype(np.float32), history=arrays["history"], **f_arrays)
            if cfg["export_step"]:
                os.replace(stmp, spath)                 # STEP first: an npz always has its STEP file
            os.replace(tmp, out)
            for suffix in ("", "_killed"):          # records of earlier rounds that failed on this design
                stale = os.path.join(cfg["out_dir"], "failures", f"{split}_{family}_{idx:05d}{suffix}.txt")
                if os.path.exists(stale):
                    os.remove(stale)
            return {"task": task, "status": "ok", "t": metrics["t_total"], "valid": metrics["valid"],
                    "attempt": attempt}
        except Rejected as ex:
            rejections.append(str(ex))
            last_err = f"Rejected: {ex}"
        except Exception as ex:  # noqa: BLE001
            fr = traceback.extract_tb(ex.__traceback__)[-1]
            rejections.append(f"{type(ex).__name__}: {ex} [{os.path.basename(fr.filename)}:{fr.lineno}]")
            last_err = f"{type(ex).__name__}: {ex}\n{traceback.format_exc()}"
    fdir = os.path.join(cfg["out_dir"], "failures")
    os.makedirs(fdir, exist_ok=True)
    with open(os.path.join(fdir, f"{split}_{family}_{idx:05d}.txt"), "w") as f:
        f.write(last_err or "unknown")
    return {"task": task, "status": "failed", "error": (last_err or "")[:300]}


# ------------------------------------------------------------------------------------------------ process pool
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")


def _limit_child_threads():
    """Set before the pool starts: spawned workers import NumPy/SciPy (and their BLAS pools) while unpickling the
    initializer, so settings made inside the initializer come too late for those libraries."""
    for k in THREAD_VARS:
        os.environ[k] = "1"


def _worker_init(gpu_slots=None):
    """Single-threaded numerics per worker. The first workers that find a free GPU slot run JAX on the GPU, the
    others on one CPU core (JAX is imported only after this, inside generate_one)."""
    for k in THREAD_VARS:
        os.environ[k] = "1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    backend = "cpu"
    if gpu_slots is not None:
        with gpu_slots.get_lock():
            if gpu_slots.value > 0:
                gpu_slots.value -= 1
                backend = "cuda"
                devices = [d for d in os.environ.get("TRIMFIELDS_GPU_DEVICES", "").split(",") if d.strip()]
                if len(devices) > 1:                          # several GPUs: round-robin over the workers
                    os.environ["CUDA_VISIBLE_DEVICES"] = devices[gpu_slots.value % len(devices)]
    os.environ["JAX_PLATFORMS"] = backend
    if backend == "cpu":
        os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


def _task_name(task):
    split, family, idx = task
    return f"{split}_{family}_{idx:05d}"


def _run_task(args):
    """One design under a watchdog: faulthandler writes the stack and kills the worker when the design exceeds
    task_timeout (this also works when OpenCASCADE hangs inside C++), and records the stack of a crash."""
    import faulthandler
    task, cfg = args
    wd = os.path.join(cfg["out_dir"], "watchdog")
    os.makedirs(wd, exist_ok=True)
    fn = os.path.join(wd, _task_name(task) + ".txt")
    with open(fn, "w") as fh:
        faulthandler.enable(file=fh)
        faulthandler.dump_traceback_later(cfg.get("task_timeout", 3600), exit=True, file=fh)
        try:
            r = generate_one(task, cfg)
        except BaseException as ex:  # noqa: BLE001
            r = {"task": task, "status": "failed", "error": f"{type(ex).__name__}: {ex}"[:300]}
        finally:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.disable()
    try:
        os.remove(fn)
    except FileNotFoundError:
        pass
    return r


def _pending(cfg, tasks, skip):
    todo = [t for t in tasks if t not in skip and not is_done(cfg, t)]
    order = {"shell": 0, "shell_ood": 0, "rand_shell": 1, "rand_shell_ood": 1}   # longest designs first
    return sorted(todo, key=lambda t: (order.get(t[0], 2), t[2], t[0], t[1]))


IDENTITY_EXCLUDE = ("out_dir", "attempt_offset", "task_timeout")


def check_identity(cfg):
    """Resuming is only allowed with the configuration that produced the designs already in out_dir."""
    ident = {k: v for k, v in cfg.items() if k not in IDENTITY_EXCLUDE}
    ident["counts"] = None                                     # more or fewer designs per family are fine
    path = os.path.join(cfg["out_dir"], "config.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old != json.loads(json.dumps(ident)):
            diff = sorted(k for k in set(old) | set(ident) if old.get(k) != json.loads(json.dumps(ident)).get(k))
            raise RuntimeError(f"{cfg['out_dir']} holds designs of another configuration (differences: {diff}); "
                               "use a new output directory or restore the configuration")
    else:
        with open(path, "w") as f:
            json.dump(ident, f, indent=1)


def _progress_bar(total):
    """Plain-text progress bar. Jupyter widget bars (tqdm.auto) break with old ipywidgets and show nothing in
    headless nbconvert runs, whereas text output works everywhere; progress.log holds the full record."""
    import sys
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc="designs", smoothing=0.01, mininterval=60, file=sys.stdout, ascii=True)
    except Exception:  # noqa: BLE001
        return None


def run_all(cfg, tasks, n_workers, progress=True, max_restarts=200, n_gpu_workers=0):
    """Generate every missing design with a spawn-based process pool; returns the list of task results.
    n_gpu_workers of the n_workers processes run the JAX part on the GPU.

    Resumable: designs already on disk are skipped. If a worker dies (watchdog timeout or crash), the pool is
    restarted; the design that caused it is identified from its watchdog file and skipped in this call."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool
    tasks = [tuple(t) for t in tasks]
    os.makedirs(cfg["out_dir"], exist_ok=True)
    check_identity(cfg)
    _limit_child_threads()
    wd = os.path.join(cfg["out_dir"], "watchdog")
    os.makedirs(wd, exist_ok=True)
    for f in os.listdir(wd):                                   # stale files of an interrupted earlier run
        os.remove(os.path.join(wd, f))
    skip, suspects, results = set(), {}, []
    todo = _pending(cfg, tasks, skip)
    bar = None
    if progress and todo:
        bar = _progress_bar(len(todo))
    log = open(os.path.join(cfg["out_dir"], "progress.log"), "a")
    ctx = mp.get_context("spawn")
    restarts = 0
    while todo:
        n_before = len(results)
        try:
            slots = ctx.Value("i", int(n_gpu_workers))
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=_worker_init,
                                     initargs=(slots,)) as pool:
                futs = [pool.submit(_run_task, (t, cfg)) for t in todo]
                for f in as_completed(futs):
                    r = f.result()
                    r["task"] = tuple(r["task"])
                    results.append(r)
                    skip.add(r["task"])
                    log.write(json.dumps({"task": list(r["task"]), "status": r["status"], "t": r.get("t"),
                                          "attempt": r.get("attempt"), "time": time.time()}) + "\n")
                    log.flush()
                    if bar is not None:
                        bar.update(1)
                        bar.set_postfix(failed=sum(1 for x in results if x["status"] != "ok"), refresh=False)
        except BrokenProcessPool:
            restarts += 1
            if len(results) == n_before and not os.listdir(wd):
                raise RuntimeError("worker processes die before running any design (check that the trimfields "
                                   "package is importable and the environment variables are valid)")
            fdir = os.path.join(cfg["out_dir"], "failures")
            os.makedirs(fdir, exist_ok=True)
            by_name = {_task_name(t): t for t in tasks}
            for f in os.listdir(wd):
                p = os.path.join(wd, f)
                t = by_name.get(f[:-4])
                if t is None:
                    os.remove(p)
                    continue
                text = open(p).read()
                suspects[t] = suspects.get(t, 0) + 1
                if text.strip() or suspects[t] >= 3:            # watchdog stack or repeated crash: give up here
                    os.replace(p, os.path.join(fdir, f"{f[:-4]}_killed.txt"))
                    skip.add(t)
                    r = {"task": t, "status": "killed"}
                    results.append(r)
                    log.write(json.dumps({"task": list(t), "status": "killed", "time": time.time()}) + "\n")
                    if bar is not None:
                        bar.update(1)
                else:
                    os.remove(p)
            log.flush()
            if restarts > max_restarts:
                raise RuntimeError("too many worker crashes; see failures/ and progress.log")
        todo = _pending(cfg, tasks, skip)
    if bar is not None:
        bar.close()
    log.close()
    return results


def generate_until_complete(cfg, n_workers, rounds=10, progress=True, n_gpu_workers=0):
    """run_all over all tasks, then retry rounds for designs still missing. Every round draws new problems: the
    attempt offset is stored in out_dir/rounds.json, so a rerun of the notebook continues with unused seeds."""
    tasks = task_list(cfg)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    state_path = os.path.join(cfg["out_dir"], "rounds.json")
    state = {"next_offset": 0, "history": []}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
    for _ in range(rounds):
        missing = [t for t in tasks if not is_done(cfg, t)]
        if not missing:
            break
        c = copy.deepcopy(cfg)
        c["attempt_offset"] = state["next_offset"]
        state["next_offset"] += cfg["max_attempts"]
        with open(state_path, "w") as f:                       # reserve the seeds before using them
            json.dump(state, f, indent=1)
        res = run_all(c, missing, n_workers, progress=progress, n_gpu_workers=n_gpu_workers)
        left = sum(1 for t in tasks if not is_done(cfg, t))
        state["history"].append({"offset": c["attempt_offset"], "attempted": len(res), "missing": left,
                                 "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        with open(state_path, "w") as f:
            json.dump(state, f, indent=1)
        print(f"round with attempt offset {c['attempt_offset']}: {len(res)} designs attempted, {left} missing",
              flush=True)
    return state["history"]


# ------------------------------------------------------------------------------------------------ consolidation
def _load(path):
    d = np.load(path, allow_pickle=False)
    return json.loads(str(d["problem"])), json.loads(str(d["metrics"])), d["x"], d["design_field"]


def _cond_job(path):
    from .conditioning import conditioning
    prob, _, _, df = _load(path)
    return conditioning(prob).astype(np.float16), df.astype(np.float16)


INDEX_COLUMNS = ("row", "split", "kind", "family", "idx", "test", "vstar", "J_host", "J_simp", "J_thr", "J_fit",
                 "J_ref", "J_final", "V_final", "ref_factor", "J_fine", "V_fine", "final_source", "loops_final",
                 "vm_max_loops", "attempt", "valid", "step_valid", "t_total", "path")


def consolidate(cfg, n_workers=8):
    """Index tables, canonical design tensors, conditioning and design fields, and splits for the planar and shell
    sets (optimized and random designs together; the `kind` and `split` columns tell them apart)."""
    import csv
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    out = os.path.join(cfg["out_dir"], "consolidated")
    os.makedirs(out, exist_ok=True)
    groups = {"planar": PLANAR_SPLITS, "shell": SHELL_SPLITS}
    summary = {}
    for name, splits in groups.items():
        paths, rows, X = [], [], []
        for split in splits:
            for family in cfg["counts"].get(split, {}):
                for i in range(cfg["counts"][split][family]):
                    p = design_path(cfg, split, family, i)
                    if not is_done(cfg, (split, family, i)):
                        continue
                    prob, met, x, _ = _load(p)
                    if met.get("J_host") is None:            # a design of an earlier version of the pipeline
                        raise RuntimeError(f"{p} lacks J_host: generated before trimfields 2.0.0; delete and regenerate")
                    rows.append({"row": len(rows), "split": sub_split(cfg, split, i), "kind": met["kind"],
                                 "family": family, "idx": i, "test": met["test"], "vstar": met["vstar"],
                                 "J_host": met.get("J_host"),
                                 "J_simp": met.get("simp", {}).get("J_simp"), "J_thr": met.get("J_thr"),
                                 "J_fit": met.get("J_fit"), "J_ref": met.get("J_ref"), "J_final": met["J_final"],
                                 "V_final": met["V_final"], "ref_factor": met.get("ref_factor"), "J_fine": met.get("J_fine"),
                                 "V_fine": met.get("V_fine"),
                                 "final_source": met["final_source"], "loops_final": met["loops_final"],
                                 "vm_max_loops": met.get("vm_max_loops"), "attempt": met["attempt"],
                                 "valid": met["valid"], "step_valid": met.get("step", {}).get("valid"),
                                 "t_total": met["t_total"], "path": os.path.relpath(p, cfg["out_dir"])})
                    paths.append(p)
                    X.append(x)
        if not rows:
            continue
        with open(os.path.join(out, f"{name}_index.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(INDEX_COLUMNS))
            w.writeheader()
            w.writerows(rows)
        np.save(os.path.join(out, f"{name}_X.npy"), np.stack(X).astype(np.float32))
        _limit_child_threads()
        cond = design = None
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("spawn"),
                                 initializer=_worker_init) as pool:
            for i, (c, d) in enumerate(pool.map(_cond_job, paths, chunksize=16)):     # results arrive in order
                if cond is None:
                    cond = np.lib.format.open_memmap(os.path.join(out, f"{name}_cond.npy"), mode="w+",
                                                     dtype=np.float16, shape=(len(paths),) + c.shape)
                    design = np.lib.format.open_memmap(os.path.join(out, f"{name}_design.npy"), mode="w+",
                                                       dtype=np.float16, shape=(len(paths),) + d.shape)
                cond[i], design[i] = c, d
        cond.flush()
        design.flush()
        del cond, design
        splits_idx = {}
        for r in rows:
            splits_idx.setdefault(r["split"], []).append(r["row"])
        with open(os.path.join(out, f"{name}_splits.json"), "w") as f:
            json.dump(splits_idx, f)
        summary[name] = {"designs": len(rows), "splits": {k: len(v) for k, v in splits_idx.items()},
                         "optimized": sum(1 for r in rows if r["kind"] == "optimized"),
                         "random": sum(1 for r in rows if r["kind"] == "random"),
                         "with_J_fine": sum(1 for r in rows if r["J_fine"] is not None),
                         "valid_fraction": float(np.mean([r["valid"] for r in rows]))}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def smoke_config(cfg, n_per_family=1, maxit=40):
    """Small configuration for a quick end-to-end check of the whole pipeline."""
    c = copy.deepcopy(cfg)
    c["out_dir"] = os.path.join(cfg["out_dir"], "_smoke")
    c["counts"] = {s: {f: n_per_family for f in fams} for s, fams in cfg["counts"].items()}
    c["refine"]["maxit"] = maxit
    c["refine"]["min_iters"] = maxit // 2
    c["simp"]["maxit"] = 80
    c["simp_shell"]["maxit"] = 120
    return c
