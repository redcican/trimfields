"""Random design problems: host, supports, loads, passive regions and target volume fraction.

Planar loads are edge tractions (or bolt bearing loads for bolted); shell loads are surface loads.
OOD-load split: in-distribution problems never centre a load in the band s in (0.42, 0.58) of edge v1;
OOD-load problems always do.

TrimFields states every boundary condition for the continuum, so that mesh-free solvers and the isogeometric
reference solve the same problem: a roller is u . n = 0 on an edge whose physical normal is constant and aligned
with a coordinate axis (fixed_component), a bolt support is u = 0 on the ring annulus around the hole, and a bolt
load is a bearing traction on the hole boundary with the cosine law of a rigid pin (resultant = mag * dir).
The random designs of the `rand` splits are admissible loop sets without optimization (random_loops).
"""
import math

import numpy as np

from .hosts import SHELL, make_host
from .mechanics import EDGES, edge_cps, edge_load, surface_load

OOD_BAND = (0.42, 0.58)
EDGE_LIST = ("u0", "u1", "v0", "v1")


def _unit(v):
    v = np.asarray(v, float)
    return (v / np.linalg.norm(v)).tolist()


def _edge_segment(rng, length_range, avoid=(), band=None, force_band=False, edge=None):
    for _ in range(200):
        e = edge or EDGE_LIST[rng.integers(4)]
        ln = rng.uniform(*length_range)
        if force_band:
            c = rng.uniform(*band)
        else:
            c = rng.uniform(ln / 2, 1 - ln / 2)
            if band is not None and e == "v1" and band[0] < c < band[1]:
                continue
        s0, s1 = c - ln / 2, c + ln / 2
        if all(not (a_e == e and s0 < a_s1 + 0.08 and s1 > a_s0 - 0.08) for a_e, a_s0, a_s1 in avoid):
            return e, float(max(0.0, s0)), float(min(1.0, s1))
    raise RuntimeError("no admissible edge segment")


def sample_problem(family, rng, split="id"):
    host = make_host(family, rng)
    prob = {"family": family, "split": split, "host": host.to_dict(), "vstar": float(rng.uniform(0.25, 0.6)),
            "supports": [], "loads": [], "passive": []}
    if family in SHELL:
        _shell_bc(prob, host, rng)
    elif family == "bolted":
        _bolt_hole_bc(prob, host, rng)
    elif family == "ell":
        _l_shape_bc(prob, rng)
    else:
        _planar_bc(prob, host, rng, split)
    return prob


def roller_edges(host, n=16, tol=1e-9):
    """Edges whose physical outward normal is constant along the edge and aligned with a coordinate axis, with the
    index of the displacement component that a roller on that edge fixes. Only such edges carry rollers, so that
    u . n = 0 is a condition on one Cartesian component, exact for the isogeometric and for continuum solvers."""
    out = {}
    s = (np.arange(n) + 0.5) / n
    for e, (axis, val) in EDGES.items():
        u, v = (np.full(n, val), s) if axis == 0 else (s, np.full(n, val))
        _, J = host.jac(u, v)
        t = J[:, :2, 1 - axis]
        nrm = np.stack([t[:, 1], -t[:, 0]], 1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        if np.abs(nrm - nrm[0]).max() > tol:
            continue
        comp = int(np.argmax(np.abs(nrm[0])))
        if abs(nrm[0, 1 - comp]) > tol:
            continue
        out[e] = comp
    return out


def _planar_bc(prob, host, rng, split):
    r = rng.uniform()
    avoid = []
    rollers = roller_edges(host) if host.dim == 2 else {}
    if r < 0.45:                                           # cantilever: one edge clamped
        e = EDGE_LIST[rng.integers(4)]
        prob["supports"].append({"kind": "edge", "edge": e, "s0": 0.0, "s1": 1.0, "dofs": "all"})
        avoid.append((e, 0.0, 1.0))
    elif r < 0.75 or not rollers:                          # two pinned segments (also where no edge admits a roller)
        for _ in range(2):
            e, s0, s1 = _edge_segment(rng, (0.05, 0.15), avoid)
            prob["supports"].append({"kind": "edge", "edge": e, "s0": s0, "s1": s1, "dofs": "all"})
            avoid.append((e, s0, s1))
    else:                                                  # roller on a full straight edge + pinned segment
        e = sorted(rollers)[rng.integers(len(rollers))]
        prob["supports"].append({"kind": "edge", "edge": e, "s0": 0.0, "s1": 1.0, "dofs": "normal",
                                 "fixed_component": rollers[e]})
        avoid.append((e, 0.0, 1.0))
        e2, s0, s1 = _edge_segment(rng, (0.05, 0.15), avoid)
        prob["supports"].append({"kind": "edge", "edge": e2, "s0": s0, "s1": s1, "dofs": "all"})
        avoid.append((e2, s0, s1))
    n_loads = 1 if rng.uniform() < 0.7 else 2
    w = [1.0] if n_loads == 1 else [float(x) for x in (lambda a: (a, 1 - a))(rng.uniform(0.3, 0.7))]
    for k in range(n_loads):
        force = split == "ood_load" and k == 0
        e, s0, s1 = _edge_segment(rng, (0.03, 0.15), avoid, band=OOD_BAND, force_band=force,
                                  edge="v1" if force else None)
        ang = rng.uniform(0, 2 * math.pi)
        prob["loads"].append({"kind": "edge", "edge": e, "s0": s0, "s1": s1,
                              "dir": [math.cos(ang), math.sin(ang)], "mag": w[k]})
        avoid.append((e, s0, s1))


def _bolt_hole_bc(prob, host, rng):
    L, H = host.params["L"], host.params["H"]
    n = int(rng.integers(1, 4))
    holes = []
    for _ in range(500):
        if len(holes) == n:
            break
        r = float(rng.uniform(0.06, 0.12) * H)
        ring = 0.08 * H
        m = r + ring + 0.05 * H
        c = (float(rng.uniform(m, L - m)), float(rng.uniform(m, H - m)))
        if all(math.hypot(c[0] - h["centre"][0], c[1] - h["centre"][1]) > r + h["radius"] + 2 * ring + 0.1 * H
               for h in holes):
            holes.append({"kind": "circle", "centre": list(c), "radius": r, "ring": ring})
    prob["passive"] = holes
    ang = rng.uniform(0, 2 * math.pi)
    prob["loads"].append({"kind": "ring", "passive": len(holes) - 1, "dir": [math.cos(ang), math.sin(ang)],
                          "mag": 1.0})
    if len(holes) == 1:
        e = EDGE_LIST[rng.integers(4)]
        prob["supports"].append({"kind": "edge", "edge": e, "s0": 0.0, "s1": 1.0, "dofs": "all"})
    else:                                       # every other hole is a bolted support (no floating rings)
        for k in range(len(holes) - 1):
            prob["supports"].append({"kind": "ring", "passive": k})


def _l_shape_bc(prob, rng):
    a = float(rng.uniform(0.35, 0.6))
    prob["passive"] = [{"kind": "polygon", "uv": [[a, a], [1.2, a], [1.2, 1.2], [a, 1.2]]}]
    prob["vstar"] *= 1.0 - (1.0 - a) ** 2          # target fraction of the L-shaped design domain, not of the square
    prob["supports"].append({"kind": "edge", "edge": "v1", "s0": 0.0, "s1": a, "dofs": "all"})
    ln = float(rng.uniform(0.04, 0.12))
    c = float(rng.uniform(ln / 2 + 0.02, a - ln / 2 - 0.02))
    ang = math.radians(rng.uniform(-135, -45))
    prob["loads"].append({"kind": "edge", "edge": "u1", "s0": c - ln / 2, "s1": c + ln / 2,
                          "dir": [math.cos(ang), math.sin(ang)], "mag": 1.0})


def _shell_bc(prob, host, rng):
    r = rng.uniform()
    if r < 0.35:                                           # two opposite edges clamped
        pair = ("u0", "u1") if rng.uniform() < 0.5 else ("v0", "v1")
        for e in pair:
            prob["supports"].append({"kind": "edge", "edge": e, "s0": 0.0, "s1": 1.0, "dofs": "clamped"})
    elif r < 0.6:                                          # four corners pinned
        for e in ("u0", "u1"):
            for s0, s1 in ((0.0, 0.08), (0.92, 1.0)):
                prob["supports"].append({"kind": "edge", "edge": e, "s0": s0, "s1": s1, "dofs": "pinned"})
    elif r < 0.8:                                          # all edges pinned
        for e in EDGE_LIST:
            prob["supports"].append({"kind": "edge", "edge": e, "s0": 0.0, "s1": 1.0, "dofs": "pinned"})
    else:                                                  # one edge clamped (cantilever shell)
        e = EDGE_LIST[rng.integers(4)]
        prob["supports"].append({"kind": "edge", "edge": e, "s0": 0.0, "s1": 1.0, "dofs": "clamped"})
    # concentrated loads: perforated shells carry them through rib-like load paths (distributed pressure or
    # self-weight favours continuous membranes, for which holes are not an efficient design)
    n_loads = 1 if rng.uniform() < 0.6 else 2
    w = [1.0] if n_loads == 1 else [float(x) for x in (lambda a: (a, 1 - a))(rng.uniform(0.3, 0.7))]
    for k in range(n_loads):
        c = [float(x) for x in rng.uniform(0.2, 0.8, 2)]
        d = "normal" if rng.uniform() < 0.5 else [0.0, 0.0, -1.0]
        prob["loads"].append({"kind": "surface", "dir": d, "centre": c, "radius": float(rng.uniform(0.04, 0.08)),
                              "mag": w[k]})
    prob["vstar"] = float(rng.uniform(0.25, 0.5))


# ------------------------------------------------------------------------------------------------ passive regions
def passive_masks_cp(prob, host):
    """Control points inside passive voids (rho = 0) and passive solid rings (rho = 1)."""
    P = host.P
    void = np.zeros(len(P), bool)
    solid = np.zeros(len(P), bool)
    gu, gv = host.greville_uv()
    GU, GV = np.meshgrid(gu, gv, indexing="ij")
    uv = np.stack([GU.ravel(), GV.ravel()], 1)
    for ps in prob["passive"]:
        if ps["kind"] == "circle":
            d = np.hypot(P[:, 0] - ps["centre"][0], P[:, 1] - ps["centre"][1])
            void |= d < ps["radius"]
            solid |= (d >= ps["radius"]) & (d < ps["radius"] + ps["ring"])
        else:
            from matplotlib.path import Path
            void |= Path(np.asarray(ps["uv"])).contains_points(uv)
    return void, solid


def ring_cps(prob, host, k):
    ps = prob["passive"][k]
    d = np.hypot(host.P[:, 0] - ps["centre"][0], host.P[:, 1] - ps["centre"][1])
    return np.where((d >= ps["radius"]) & (d < ps["radius"] + ps["ring"]))[0]


# ------------------------------------------------------------------------------------------------ non-design regions
def _boundary_items(prob):
    """Edge supports/loads and patch loads; ring supports/loads are already passive solid rings."""
    edges = [x for x in prob["supports"] + prob["loads"] if x["kind"] == "edge"]
    patches = [x for x in prob["loads"] if x["kind"] == "surface" and x.get("centre") is not None]
    return edges, patches


def keep_cps(prob, host):
    """Control points kept solid in SIMP: two edge rows under every support and edge load, and the control points
    whose Greville abscissae lie in a load patch (standard non-design regions of density-based TO)."""
    keep = np.zeros(host.ncu * host.ncv, bool)
    edges, patches = _boundary_items(prob)
    for x in edges:                             # two rows: a solid strip of about 1.5 elements
        for row in (0, 1):
            keep[edge_cps(host, x["edge"], x["s0"], x["s1"], row=row)] = True
    if patches:
        gu, gv = host.greville_uv()
        GU, GV = np.meshgrid(gu, gv, indexing="ij")
        for x in patches:
            r = max(x["radius"], 0.75 / min(host.nelu, host.nelv))
            keep |= ((GU - x["centre"][0]) ** 2 + (GV - x["centre"][1]) ** 2 <= r**2).ravel()
    return keep


def keep_geometry(prob, n=96):
    """Exact protected regions in the parameter square (shapely): supported and loaded boundary segments, load
    patches and bolt-hole rings. An admissible loop interior must not overlap any of them."""
    from shapely.geometry import LineString, Point, Polygon
    edges, patches = _boundary_items(prob)
    out = []
    for x in edges:
        axis, val = EDGES[x["edge"]]
        a, b = (val, x["s0"]), (val, x["s1"])
        out.append(LineString([a, b] if axis == 0 else [a[::-1], b[::-1]]))
    for x in patches:
        out.append(Point(x["centre"]).buffer(x["radius"], quad_segs=n // 4))
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for ps in prob["passive"]:
        if ps["kind"] == "circle":
            L, H = prob["host"]["params"]["L"], prob["host"]["params"]["H"]
            ring = [np.stack([(ps["centre"][0] + r * np.cos(t)) / L, (ps["centre"][1] + r * np.sin(t)) / H], 1)
                    for r in (ps["radius"] + ps["ring"], ps["radius"])]
            out.append(Polygon(ring[0], [ring[1]]))
    return out


def keep_samples(keep_geoms, spacing=0.01):
    """Parametric sample points of the non-design set: points along protected boundary segments and on the
    boundary and interior of protected patches and rings (used by the keep-region penalty of the refinement)."""
    pts = []
    for g in keep_geoms:
        if g.geom_type == "LineString":
            n = max(2, int(np.ceil(g.length / spacing)) + 1)
            pts.append(np.array([g.interpolate(t, normalized=True).coords[0] for t in np.linspace(0, 1, n)]))
            continue
        for ring in [g.exterior] + list(g.interiors):
            n = max(8, int(np.ceil(ring.length / spacing)))
            pts.append(np.array([ring.interpolate(t, normalized=True).coords[0] for t in np.linspace(0, 1, n,
                                                                                                  endpoint=False)]))
        x0, y0, x1, y1 = g.bounds
        X, Y = np.meshgrid(np.arange(x0, x1, spacing), np.arange(y0, y1, spacing), indexing="ij")
        cand = np.stack([X.ravel(), Y.ravel()], 1)
        if len(cand):
            import shapely
            pts.append(cand[shapely.contains_xy(g, cand[:, 0], cand[:, 1])])
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def covers_keep(loop_poly, keep_geoms, tol=1e-7):
    """True if a loop interior (shapely polygon) overlaps a protected region by more than touching."""
    for g in keep_geoms:
        if not loop_poly.intersects(g):
            continue
        inter = loop_poly.intersection(g)
        if (inter.length if g.geom_type == "LineString" else inter.area) > tol:
            return True
    return False


def passive_discs_uv(prob, scale=1.0, n=64):
    """Hole-plus-ring discs of the bolt holes as parametric polygons (rectangle host)."""
    out = []
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for ps in prob["passive"]:
        if ps["kind"] == "circle":
            L, H = prob["host"]["params"]["L"], prob["host"]["params"]["H"]
            r = scale * (ps["radius"] + ps["ring"])
            out.append(np.stack([(ps["centre"][0] + r * np.cos(t)) / L, (ps["centre"][1] + r * np.sin(t)) / H], 1))
    return out


# ------------------------------------------------------------------------------------------------ random designs
def random_loops(prob, host, nel, rng, n_cp=16, max_loops=8, w_min_h=1.5, keep_geoms=(), tries=400):
    """Admissible loop set without optimization (the `rand` splits): 1 to max_loops closed loops, each an ellipse
    with a low-order radial perturbation (a blob), with radii of 1.5 to 8 elements, pairwise clearance of w_min and
    no overlap with the non-design set. Loops near the patch boundary may protrude beyond the square (notches)."""
    from shapely.geometry import Polygon, box

    from .loops import fit_periodic, sample_loop
    L = np.array([host.Lu, host.Lv])
    h = math.sqrt(host.area / (nel[0] * nel[1]))
    w_min = w_min_h * h
    el_area = 1.0 / (nel[0] * nel[1])
    K = int(rng.integers(1, max_loops + 1))
    loops, polys = [], []
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    for _ in range(tries):
        if len(loops) == K:
            break
        c = rng.uniform(0.04, 0.96, 2) * L
        r0 = rng.uniform(1.5, 8.0) * h
        asp = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
        a, b = r0 * math.sqrt(asp), r0 / math.sqrt(asp)
        amp, ph = rng.uniform(0.0, 0.25, 3), rng.uniform(0.0, 2 * np.pi, 3)
        rad = 1.0 + sum(amp[i] * np.cos((i + 2) * t + ph[i]) for i in range(3))
        th = rng.uniform(0.0, 2 * np.pi)
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        pts = c + (R @ np.stack([a * rad * np.cos(t), b * rad * np.sin(t)])).T
        C = fit_periodic(pts, n_cp) / L
        poly_uv = Polygon(sample_loop(C, 20)).buffer(0)
        if poly_uv.is_empty or poly_uv.intersection(box(0, 0, 1, 1)).area < 2 * el_area:
            continue
        if keep_geoms and covers_keep(poly_uv, keep_geoms):
            continue
        poly_z = Polygon(sample_loop(C, 20) * L).buffer(0)
        if any(poly_z.distance(p) < w_min for p in polys):
            continue
        loops.append(C)
        polys.append(poly_z)
    return loops


# ------------------------------------------------------------------------------------------------ boundary conditions
def bearing_load(model, prob, k, direction, magnitude, n=256):
    """Consistent nodal forces of the bearing traction of a rigid pin in bolt hole k (rectangle host): the radial
    traction p0 cos(phi) e_r on the half of the hole boundary facing the load direction, with p0 = 2 F / (pi r), so
    that the resultant is magnitude * direction; the line integral runs over the exact ellipse of the hole in the
    parameter square."""
    host = model.host
    ps = prob["passive"][k]
    L, H = prob["host"]["params"]["L"], prob["host"]["params"]["H"]
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    phi = (np.arange(n) + 0.5) * 2 * np.pi / n
    er = np.stack([np.cos(phi), np.sin(phi)], 1)
    cosw = np.maximum(er @ d, 0.0)
    r = ps["radius"]
    ds = 2 * np.pi * r / n
    traction = (2 * magnitude / (np.pi * r)) * cosw[:, None] * er
    x = ps["centre"][0] + r * np.cos(phi)
    y = ps["centre"][1] + r * np.sin(phi)
    b = host.basis(x / L, y / H, nd=0)
    F = np.zeros(model.ndof)
    kd = model.ndpc
    for c in range(kd):
        np.add.at(F, kd * b["ien"].ravel() + c, (b["R"] * (traction[:, c] * ds)[:, None]).ravel())
    res = np.array([F[c::kd].sum() for c in range(kd)])            # quadrature of the cosine law: rescale exactly
    F *= magnitude / np.linalg.norm(res)
    return F


def apply_bc(prob, model):
    host, k = model.host, model.ndpc
    fixed = []
    for s in prob["supports"]:
        if s["kind"] == "ring":                           # bolt support: u = 0 on the ring annulus
            cps = ring_cps(prob, host, s["passive"])
            fixed += [k * cps + d for d in range(k)]
            continue
        cps = edge_cps(host, s["edge"], s["s0"], s["s1"])
        if s["dofs"] == "all" or s["dofs"] == "pinned":
            fixed += [k * cps + d for d in range(k)]
        elif s["dofs"] == "clamped":
            cps1 = edge_cps(host, s["edge"], s["s0"], s["s1"], row=1)
            fixed += [k * c + d for c in (cps, cps1) for d in range(k)]
        elif s["dofs"] == "normal":                      # roller: u . n = 0 on an axis-aligned straight edge
            comp = s.get("fixed_component")
            if comp is None:
                comp = roller_edges(host)[s["edge"]]
            fixed.append(k * cps + int(comp))
    F = np.zeros(model.ndof)
    for ld in prob["loads"]:
        if ld["kind"] == "edge":
            F += edge_load(model, ld["edge"], ld["s0"], ld["s1"], ld["dir"], ld["mag"])
        elif ld["kind"] == "ring":
            F += bearing_load(model, prob, ld["passive"], ld["dir"], ld["mag"])
        else:
            F += surface_load(model, ld["dir"] if isinstance(ld["dir"], str) else ld["dir"], ld["mag"],
                              ld["centre"], ld["radius"])
    model.set_bc(np.concatenate([np.atleast_1d(f) for f in fixed]) if fixed else np.zeros(0, int), F)
    return model
