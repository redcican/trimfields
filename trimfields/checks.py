"""Numerical checks run before data generation: basis functions, analytic and benchmark solutions, design
gradients, cut-cell quadrature and STEP export. Each check returns a dict with its measured error and a pass flag."""
import math
import os
import tempfile

import numpy as np


def check_basis(seed=0):
    """Partition of unity and finite-difference derivatives of a rational host (plate with a hole)."""
    from .hosts import make_host
    host = make_host("plate", np.random.default_rng(seed)).build(12, 6)
    rng = np.random.default_rng(seed + 1)
    u, v = rng.uniform(0.05, 0.95, 200), rng.uniform(0.05, 0.95, 200)
    b = host.basis(u, v, nd=1)
    pu = float(np.abs(b["R"].sum(1) - 1).max())
    eps = 1e-6
    bp, bm = host.basis(u + eps, v, nd=0), host.basis(u - eps, v, nd=0)
    same = np.all(bp["ien"] == b["ien"], 1) & np.all(bm["ien"] == b["ien"], 1)
    fd = (bp["R"] - bm["R"]) / (2 * eps)
    d_err = float(np.abs(fd[same] - b["Ru"][same]).max())
    return {"partition_of_unity": pu, "dR_du_fd": d_err, "pass": pu < 1e-12 and d_err < 1e-6}


def check_tension():
    """Plane-stress bar in uniaxial tension: compliance F^2 L / (E H t) = 2 for L = 2, H = 1."""
    from .hosts import BEZ, Host, rectangle
    from .mechanics import Model, edge_cps, edge_load
    P, W = rectangle(2.0, 1.0)
    h = Host("rect", {}, BEZ, BEZ, P, W).build(20, 10)
    m = Model.on_grid(h, 3)
    F = edge_load(m, "u1", 0, 1, (1, 0), 1.0)
    m.set_bc(np.r_[2 * edge_cps(h, "u0", 0, 1), 1], F)
    J, U = m.solve(np.ones(m.u.size))
    err = abs(J - 2.0) / 2.0
    e_sum = abs(m.energy_density(U).sum() / J - 1.0)
    return {"J": J, "rel_err": err, "energy_sum_err": e_sum, "pass": err < 1e-9 and e_sum < 1e-9}


def check_scordelis_lo(n=24):
    """Scordelis-Lo roof (Kirchhoff-Love shell): vertical deflection at the free-edge midpoint, reference 0.3006."""
    from .basis import gauss_grid
    from .hosts import BEZ, Host, arc3
    from .mechanics import Model, edge_cps, surface_load
    R, L, t = 25.0, 50.0, 0.25
    A, w = arc3(math.radians(80), R, start=math.radians(50))
    P = [[(A[i, 0], y, A[i, 1]) for y in (0, L / 2, L)] for i in range(3)]
    W = [[w[i]] * 3 for i in range(3)]
    h = Host("cyl", {}, BEZ, BEZ, P, W).build(n, n)
    m = Model(h, *gauss_grid(h.Uu, h.Uv, 3)[:4], thickness=t, nu=0.0)
    F = surface_load(m, (0, 0, -1), 90 * h.area)
    fixed = []
    for edge in ("v0", "v1"):                                  # rigid diaphragms: u_x = u_z = 0
        cps = edge_cps(h, edge, 0, 1)
        fixed += list(3 * cps) + list(3 * cps + 2)
    fixed.append(1)                                            # axial rigid-body mode (no axial load)
    m.set_bc(np.array(fixed), F)
    _, U = m.solve(np.full(m.u.size, 4.32e8))
    b = h.basis(np.array([0.0]), np.array([0.5]), nd=0)
    wz = -float((b["R"][0] * U[3 * b["ien"][0] + 2]).sum())
    err = abs(wz - 0.3006) / 0.3006
    return {"w_mid": wz, "reference": 0.3006, "rel_err": err, "pass": err < 0.01}


def _circle_loops(centres, radii, n_cp=16):
    from .loops import fit_periodic
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    return [fit_periodic(np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], 1), n_cp)
            for c, r in zip(centres, radii)]


def _cantilever(n_el=600):
    from .hosts import Host, make_host
    from .problems import apply_bc
    host = make_host("rect", np.random.default_rng(3))
    prob = {"family": "rect", "split": "id", "host": host.to_dict(), "vstar": 0.5, "passive": [], "thickness": None,
            "supports": [{"kind": "edge", "edge": "u0", "s0": 0.0, "s1": 1.0, "dofs": "all"}],
            "loads": [{"kind": "edge", "edge": "u1", "s0": 0.4, "s1": 0.6, "dir": [0.0, -1.0], "mag": 1.0}]}
    host = Host.from_dict(prob["host"])
    host.build(*host.mesh_for(n_el))
    return prob, host, apply_bc


def check_gradient(seed=0):
    """Design gradient of the compliance with respect to the loop control points against central differences."""
    from .indicator import Indicator
    from .mechanics import Model
    from .refine import Refiner
    from .pipeline import CONFIG
    prob, host, apply_bc = _cantilever()
    m4 = apply_bc(prob, Model.on_grid(host, 4))
    ind = Indicator(m4, prob, 16, 16)
    ref = Refiner(m4, ind, CONFIG["refine"])
    Cz, act = ref._slots(_circle_loops([(0.3, 0.5), (0.68, 0.45)], [0.12, 0.09]))
    J, _, g = ref._J_and_grad(Cz, act)
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(3):
        D = rng.normal(size=Cz.shape) * act[:, None, None]
        D /= np.linalg.norm(D)
        eps = 1e-3 * ind.h
        Jp = ref._J_and_grad(Cz + eps * D, act)[0]
        Jm = ref._J_and_grad(Cz - eps * D, act)[0]
        fd, an = (Jp - Jm) / (2 * eps), float(np.sum(g * D))
        errs.append(abs(fd - an) / max(abs(fd), 1e-12))
    return {"J": J, "max_rel_err": max(errs), "pass": max(errs) < 1e-4}


def check_cut_cell():
    """Crisp volume fraction of a two-hole design against the exact area of the sampled loops."""
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    from .evaluate import Evaluator
    from .loops import sample_loop
    prob, host, _ = _cantilever()
    loops = _circle_loops([(0.3, 0.5), (0.68, 0.45)], [0.12, 0.09])
    ev = Evaluator(prob, (host.nelu, host.nelv), None, 2)
    r = ev.evaluate(ev.loop_material(loops))
    holes = unary_union([Polygon(sample_loop(C, 400)) for C in loops]).intersection(box(0, 0, 1, 1))
    V_exact = 1.0 - holes.area                  # rectangle host: parametric and physical fractions agree
    err = abs(r["V"] - V_exact)
    return {"V": r["V"], "V_exact": V_exact, "abs_err": err, "cut_elements": r["cut_elements"], "pass": err < 2e-3}


def check_step():
    """STEP export with pcurves: B-rep validity after reading the file back, and face area against the design."""
    from .evaluate import Evaluator
    from .export_step import HAVE_OCP, export_design
    if not HAVE_OCP:
        return {"pass": False, "error": "OCP not importable"}
    prob, host, _ = _cantilever()
    loops = _circle_loops([(0.3, 0.5), (0.68, 0.45)], [0.12, 0.09])
    ev = Evaluator(prob, (host.nelu, host.nelv), None, 2)
    V = ev.evaluate(ev.loop_material(loops))["V"]
    with tempfile.TemporaryDirectory() as d:
        st = export_design(prob, loops, os.path.join(d, "check.step"))
    err = abs(st["area"] - V * host.area) / (V * host.area)
    return {"valid": st["valid"], "faces": st["faces"], "area_rel_err": err, "pass": st["valid"] and err < 2e-3}


def check_solvers():
    """PARDISO and SuperLU give the same compliance (only when pypardiso is installed)."""
    from . import mechanics as M
    if not M.HAVE_PARDISO:
        return {"pass": True, "note": "pypardiso not installed; SuperLU is used"}
    prob, host, apply_bc = _cantilever()
    m = apply_bc(prob, M.Model.on_grid(host, 3))
    E = np.random.default_rng(0).uniform(0.1, 1.0, m.u.size)
    use = M.USE_PARDISO
    try:
        M.USE_PARDISO = True
        J1 = m.solve(E)[0]
        M.USE_PARDISO = False
        J2 = m.solve(E)[0]
    finally:
        M.USE_PARDISO = use
    err = abs(J1 - J2) / abs(J2)
    return {"J_pardiso": J1, "J_superlu": J2, "rel_diff": err, "pass": err < 1e-8}


def check_bearing():
    """Bearing load of a bolt hole: the resultant equals magnitude * direction, the traction acts on the half of the
    hole facing the load, and the compliance equals the energy of the solution (plate with one loaded hole)."""
    from .hosts import make_host
    from .mechanics import Model
    from .problems import apply_bc, bearing_load
    host = make_host("bolted", np.random.default_rng(5))
    L, H = host.params["L"], host.params["H"]
    prob = {"family": "bolted", "split": "id", "host": host.to_dict(), "vstar": 0.5, "thickness": None,
            "passive": [{"kind": "circle", "centre": [0.5 * L, 0.5 * H], "radius": 0.1 * H, "ring": 0.08 * H}],
            "supports": [{"kind": "edge", "edge": "u0", "s0": 0.0, "s1": 1.0, "dofs": "all"}],
            "loads": [{"kind": "ring", "passive": 0, "dir": [0.6, -0.8], "mag": 2.0}]}
    host.build(*host.mesh_for(600))
    m = Model.on_grid(host, 3)
    F = bearing_load(m, prob, 0, prob["loads"][0]["dir"], 2.0)
    res = np.array([F[0::2].sum(), F[1::2].sum()])
    res_err = float(np.linalg.norm(res - 2.0 * np.array([0.6, -0.8])))
    # forces are assigned only to control points whose basis functions reach the loaded half of the hole
    P = host.P
    d = np.stack([P[:, 0] - 0.5 * L, P[:, 1] - 0.5 * H], 1)
    side = d @ np.array([0.6, -0.8]) / np.maximum(np.linalg.norm(d, axis=1), 1e-300)
    loaded = np.abs(F[0::2]) + np.abs(F[1::2]) > 1e-12
    behind = float(np.sum(loaded & (side < -0.6)) / max(loaded.sum(), 1))      # share of loaded cps behind the hole
    apply_bc(prob, m)
    J, U = m.solve(np.ones(m.u.size))
    e_err = float(abs(m.energy_density(U).sum() / J - 1.0))
    return {"resultant_err": res_err, "cps_behind_hole": behind, "J": float(J), "energy_sum_err": e_err,
            "pass": res_err < 1e-10 and behind < 0.05 and e_err < 1e-9}


def check_fields():
    """Reference fields of a design: the stress and energy of fields.stress_energy_at agree with the model's own
    quadrature-point values, the loop normals are unit vectors pointing into the holes, and the design field
    reproduces the crisp material test of the evaluator."""
    from .evaluate import Evaluator
    from .fields import design_field, loop_samples, material_mask, stress_energy_at
    from .hosts import Host
    prob, host, _ = _cantilever()
    centres, radii = [(0.3, 0.5), (0.68, 0.45)], [0.12, 0.09]
    loops = _circle_loops(centres, radii)
    ev = Evaluator(prob, (host.nelu, host.nelv), None, 2)
    r = ev.evaluate(ev.loop_material(loops), keep_model=True)
    m, U = r["model"], r["U"]
    sel = np.arange(0, m.u.size, max(1, m.u.size // 500))
    _, w = stress_energy_at(m, U, m.u[sel], m.v[sel])
    w_ref = m.energy_density(U)[sel] / m.dA[sel]
    w_err = float(np.abs(w - w_ref).max() / np.abs(w_ref).max())
    ls = loop_samples(m, U, loops, 32)
    n_err = float(np.abs(np.linalg.norm(ls["normal"], axis=2) - 1.0).max())
    L = np.array([host.Lu, host.Lv])
    inward = []
    for k, (c, _) in enumerate(zip(centres, radii)):               # rectangle host: X = (Lu u, Lv v)
        to_centre = np.asarray(c) * L - ls["x"][k]
        inward.append(np.sum(to_centre / np.linalg.norm(to_centre, axis=1, keepdims=True) * ls["normal"][k], 1))
    inward = float(np.min(np.concatenate(inward)))
    df = design_field(prob, loops)
    n = df.shape[1]
    c = (np.arange(n) + 0.5) / n
    Ug, Vg = np.meshgrid(c, c, indexing="ij")
    mat = material_mask(prob, Host.from_dict(prob["host"]).build(8, 8), Ug.ravel(), Vg.ravel(), loops)
    chi_err = int(np.sum((df[0].ravel() > 0) != mat))
    sdf_sign_err = int(np.sum((df[1].ravel() > 0) != mat))
    return {"energy_err": w_err, "normal_err": n_err, "normal_inward_min": inward, "chi_mismatch": chi_err,
            "sdf_sign_mismatch": sdf_sign_err, "vm_max": float(ls["vm"].max()),
            "pass": w_err < 1e-9 and n_err < 1e-9 and inward > 0.9 and chi_err == 0 and sdf_sign_err == 0}


def check_bodyfit_core():
    """Quadratic-triangle plane-stress solver of bodyfit.py (without gmsh) on the bar in uniaxial tension of
    check_tension: compliance 2, and the edge-segment load integration recovers the exact resultant."""
    from .bodyfit import solve_face, structured_p2
    from .hosts import BEZ, Host, rectangle
    P, W = rectangle(2.0, 1.0)
    host = Host("rect", {"L": 2.0, "H": 1.0}, BEZ, BEZ, P, W)
    prob = {"family": "rect", "split": "id", "host": host.to_dict(), "vstar": 1.0, "passive": [], "thickness": None,
            "supports": [{"kind": "edge", "edge": "u0", "s0": 0.0, "s1": 1.0, "dofs": "normal", "fixed_component": 0},
                         {"kind": "edge", "edge": "u0", "s0": 0.0, "s1": 0.001, "dofs": "all"}],
            "loads": [{"kind": "edge", "edge": "u1", "s0": 0.0, "s1": 1.0, "dir": [1.0, 0.0], "mag": 1.0}]}
    nodes, tris = structured_p2(2.0, 1.0, 8, 4)
    r = solve_face(prob, nodes, tris)
    err = abs(r["J"] - 2.0) / 2.0
    prob["loads"][0].update({"s0": 0.3, "s1": 0.7})            # partial segment, not aligned with mesh edges
    r2 = solve_face(prob, nodes, tris)
    prob["loads"][0].update({"s0": 0.0, "s1": 1.0})
    nodes_i, tris_i = structured_p2(0.4, 0.3, 2, 2)              # a detached piece inside the bar: a floating island
    nodes3 = np.concatenate([nodes, nodes_i + np.array([0.8, 0.3])])
    tris3 = np.concatenate([tris, tris_i + len(nodes)])
    r3 = solve_face(prob, nodes3, tris3)
    err3 = abs(r3["J"] - 2.0) / 2.0
    return {"J": r["J"], "rel_err": err, "V": r["V"], "J_partial": r2["J"], "islands": r3["islands"],
            "J_island": r3["J"],
            "pass": err < 1e-9 and abs(r["V"] - 1.0) < 1e-12 and r2["J"] > r["J"] and r3["islands"] == 1
            and err3 < 1e-9}


def check_load_path():
    """Load-path test of the random designs: the untrimmed host has the lowest compliance of its problem, two
    holes raise it moderately, and a loop that severs the cantilever raises it by more than the rejection factor
    of the pipeline, so that admissible() reports the broken load path."""
    from .evaluate import Evaluator
    from .loops import fit_periodic
    from .pipeline import CONFIG, admissible
    prob, host, _ = _cantilever()
    ev = Evaluator(prob, (host.nelu, host.nelv), None, 2)
    J_host = ev.evaluate(ev.loop_material([]))["J"]
    holes = _circle_loops([(0.3, 0.5), (0.68, 0.45)], [0.12, 0.09])
    r_holes = ev.evaluate(ev.loop_material(holes))
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)                 # a tall ellipse across the whole height
    cut = [fit_periodic(np.stack([0.55 + 0.04 * np.cos(t), 0.5 + 0.7 * np.sin(t)], 1), 16)]
    r_cut = ev.evaluate(ev.loop_material(cut))
    factor = CONFIG["random"]["max_compliance_host"]
    verdict = admissible(cut, r_cut, r_cut["V"], (), CONFIG["vol_tol"], factor * J_host)
    return {"J_host": J_host, "J_holes_ratio": r_holes["J"] / J_host, "J_cut_ratio": r_cut["J"] / J_host,
            "verdict": verdict,
            "pass": J_host > 0 and 1.0 < r_holes["J"] / J_host < factor and r_cut["J"] / J_host > factor
            and verdict.startswith("compliance")}


def check_uncut_host():
    """Untrimmed L-shaped host whose cut-out edges lie between the classification samples of the evaluator, so
    that no element is cut: the evaluation (used for J_host of every design on such a host) must still return a
    finite compliance and the volume of the L-shaped domain. Before 2.0.1 the material test was called at no
    points and failed, which resampled the problem."""
    from .evaluate import Evaluator
    from .hosts import Host, make_host
    host = make_host("ell", np.random.default_rng(5))
    nel = host.mesh_for(600)
    h = 1.0 / (2 * nel[0])                                     # element width of the evaluator at f = 2
    a = 0.5 + 0.05 * h                                         # cut-out corner just beyond the knot line 0.5
    prob = {"family": "ell", "split": "id", "host": host.to_dict(), "vstar": 0.4, "thickness": None,
            "passive": [{"kind": "polygon", "uv": [[a, a], [1.2, a], [1.2, 1.2], [a, 1.2]]}],
            "supports": [{"kind": "edge", "edge": "v1", "s0": 0.0, "s1": a, "dofs": "all"}],
            "loads": [{"kind": "edge", "edge": "u1", "s0": 0.2, "s1": 0.3, "dir": [0.0, -1.0], "mag": 1.0}]}
    host = Host.from_dict(prob["host"])
    ev = Evaluator(prob, nel, None, 2)
    r = ev.evaluate(ev.loop_material([]))
    return {"cut_elements": r["cut_elements"], "J_host": r["J"], "V_host": r["V"],
            "pass": r["cut_elements"] == 0 and np.isfinite(r["J"]) and r["J"] > 0 and abs(r["V"] - 0.75) < 0.01}


def check_threshold_passive():
    """Passive geometry in the thresholded-density test: on a bolted plate with a density that dips far from the
    holes and is void around them (as a converged density is), the thresholded material of the evaluator is void
    inside the holes and solid on the rings regardless of the density, the level found by threshold_for_volume
    gives the target volume, the bearing load then acts on material (compliance within a factor 50 of the untrimmed
    host, that is, an intact load path), and the three material tests agree at random points:
    Evaluator.density_material with rho = 1, Evaluator.loop_material and fields.material_mask."""
    from .evaluate import Evaluator
    from .fields import material_mask
    from .hosts import Host, make_host
    from .problems import sample_problem
    rng = np.random.default_rng(5)
    prob = sample_problem("bolted", rng)
    host = Host.from_dict(prob["host"])
    host.build(*host.mesh_for(900))
    nel = (host.nelu, host.nelv)
    ev = Evaluator(prob, nel, None, 2)
    gu, gv = host.greville_uv()                              # density: a ramp in x, void within 1.3 radii of every hole
    Gu, Gv = np.meshgrid(np.asarray(gu).ravel(), np.asarray(gv).ravel(), indexing="ij")
    Pg = host.eval(Gu.ravel(), Gv.ravel())
    L, H = prob["host"]["params"]["L"], prob["host"]["params"]["H"]
    cand = np.stack(np.meshgrid(np.linspace(0.1 * L, 0.9 * L, 40), np.linspace(0.1 * H, 0.9 * H, 20)), -1).reshape(-1, 2)
    dist = np.min([np.hypot(cand[:, 0] - ps["centre"][0], cand[:, 1] - ps["centre"][1]) for ps in prob["passive"]], 0)
    xc, yc = cand[np.argmax(dist)]                           # a smooth dip of density far from the holes
    rho = 1.0 - 0.95 * np.exp(-((Pg[:, 0] - xc) ** 2 + (Pg[:, 1] - yc) ** 2) / (0.25 * L) ** 2)
    for ps in prob["passive"]:
        rho[np.hypot(Pg[:, 0] - ps["centre"][0], Pg[:, 1] - ps["centre"][1]) < 1.3 * ps["radius"]] = 0.0
    vstar = 0.9
    th = ev.threshold_for_volume(host, rho, vstar)
    r = ev.evaluate(ev.density_material(rho, th))
    J_host = ev.evaluate(ev.loop_material([]))["J"]
    u, v = rng.uniform(0, 1, 4000), rng.uniform(0, 1, 4000)
    X = host.eval(u, v)
    d = np.min([np.hypot(X[:, 0] - ps["centre"][0], X[:, 1] - ps["centre"][1]) / ps["radius"] for ps in prob["passive"]], 0)
    rings = np.zeros(len(u), bool)
    for ps in prob["passive"]:
        dd = np.hypot(X[:, 0] - ps["centre"][0], X[:, 1] - ps["centre"][1])
        rings |= (dd >= ps["radius"]) & (dd < ps["radius"] + ps["ring"])
    m_th = ev.density_material(rho, th)(u, v)
    hole_void = int(np.sum(m_th[d < 1.0]))
    ring_solid = int(np.sum(~m_th[rings]))
    m_loop = ev.loop_material([])(u, v)
    m_one = ev.density_material(np.ones_like(rho), 0.5)(u, v)
    m_fld = material_mask(prob, host, u, v, [])
    agree = int(np.sum(m_loop != m_one) + np.sum(m_loop != m_fld))
    return {"threshold": float(th), "V_thr": r["V"], "J_thr_over_host": r["J"] / J_host, "hole_points_solid": hole_void,
            "ring_points_void": ring_solid, "tests_disagree": agree,
            "pass": abs(r["V"] - vstar) < 5e-3 and np.isfinite(r["J"]) and 1.0 < r["J"] / J_host < 50.0
            and hole_void == 0 and ring_solid == 0 and agree == 0}


ALL = {"basis": check_basis, "tension": check_tension, "scordelis_lo": check_scordelis_lo,
       "gradient": check_gradient, "cut_cell": check_cut_cell, "step": check_step, "solvers": check_solvers,
       "bearing": check_bearing, "fields": check_fields, "bodyfit_core": check_bodyfit_core,
       "load_path": check_load_path, "threshold_passive": check_threshold_passive, "uncut_host": check_uncut_host}


def run_all_checks(verbose=True):
    out = {}
    for name, fn in ALL.items():
        try:
            out[name] = fn()
        except Exception as ex:  # noqa: BLE001
            out[name] = {"pass": False, "error": f"{type(ex).__name__}: {ex}"}
        if verbose:
            r = out[name]
            vals = ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items() if k != "pass")
            print(f"[{'PASS' if r['pass'] else 'FAIL'}] {name:13s} {vals}", flush=True)
    return out
