"""Reference fields stored with every design (new in TrimFields).

- stress and energy density of the isogeometric solution at arbitrary parameter points, in particular along the
  trimming loops (the cut-cell boundary stress that body-fitted and neural solvers are compared with);
- the design fields on the conditioning grid: the crisp material indicator and the signed distance to the
  trimming curves, which are the inputs of a design-conditioned neural solver.
"""
import numpy as np
from matplotlib.path import Path

from .basis import periodic_basis
from .hosts import Host
from .loops import sample_loop, signed_area

GRID = 64


# ------------------------------------------------------------------------------------------------ fields at points
def element_index(host, u, v):
    """Element index (e = iu * nelv + iv, as in basis.gauss_grid) of parameter points."""
    ku, kv = np.unique(host.Uu), np.unique(host.Uv)
    iu = np.clip(np.searchsorted(ku, u, side="right") - 1, 0, len(ku) - 2)
    iv = np.clip(np.searchsorted(kv, v, side="right") - 1, 0, len(kv) - 2)
    return iu * (len(kv) - 1) + iv


def _von_mises(s):
    return np.sqrt(np.maximum(s[:, 0] ** 2 - s[:, 0] * s[:, 1] + s[:, 1] ** 2 + 3.0 * s[:, 2] ** 2, 0.0))


def stress_energy_at(model, U, u, v):
    """Von Mises stress (unit modulus; for shells the larger value of the two outer surfaces) and the energy density
    w = eps:D:eps per unit area (membrane plus bending for shells) of the solution U at parameter points."""
    u = np.clip(np.asarray(u, float), 1e-12, 1 - 1e-12)
    v = np.clip(np.asarray(v, float), 1e-12, 1 - 1e-12)
    e = element_index(model.host, u, v)
    if model.shell:
        Bm, Bb, _ = model._shell_B(u, v, e)
        terms = [(Bm, model.thickness), (Bb, model.thickness**3 / 12.0)]
    else:
        B, _ = model._plane_B(u, v, e)
        terms = [(B, 1.0)]
    ue = U[model.edof][e]
    eps = [np.einsum("mik,mk->mi", B, ue) for B, _ in terms]
    sig = [np.einsum("ij,mj->mi", model.D0, ep) for ep in eps]
    w = sum(f * np.sum(ep * sg, 1) for (_, f), ep, sg in zip(terms, eps, sig))
    if model.shell:
        z = 0.5 * model.thickness
        vm = np.maximum(_von_mises(sig[0] + z * sig[1]), _von_mises(sig[0] - z * sig[1]))
    else:
        vm = _von_mises(sig[0])
    return vm, w


def loop_samples(model, U, loops_uv, n=64, prob=None):
    """n samples per loop: parameter point, physical point, unit normal pointing into the hole, arc-length weight,
    von Mises stress, energy density and a flag for samples on a free boundary of the face: inside the parameter
    square and, when `prob` is given, outside every passive region (loop parts beyond the square or inside a bolt
    hole, a ring or a cut-out bound no material). Arrays of shape (K, n, ...)."""
    host = model.host
    K, dim = len(loops_uv), host.dim
    out = {"uv": np.zeros((K, n, 2)), "x": np.zeros((K, n, dim)), "normal": np.zeros((K, n, dim)),
           "ds": np.zeros((K, n)), "vm": np.zeros((K, n)), "w": np.zeros((K, n)), "inside": np.zeros((K, n), bool)}
    r = np.arange(n) / n
    for k, C in enumerate(loops_uv):
        C = np.asarray(C, float)
        Bp, Bd = periodic_basis(len(C), r), periodic_basis(len(C), r, der=1)
        uv, tuv = Bp @ C, Bd @ C
        inside = np.all((uv >= 0.0) & (uv <= 1.0), 1)
        uvc = np.clip(uv, 1e-9, 1 - 1e-9)
        if prob is not None:
            inside &= ~in_passive(prob, host, uvc[:, 0], uvc[:, 1])
        _, J = host.jac(uvc[:, 0], uvc[:, 1])
        X = host.eval(uvc[:, 0], uvc[:, 1])
        t = np.einsum("mdi,mi->md", J, tuv)
        tn = np.linalg.norm(t, axis=1, keepdims=True)
        t = t / np.maximum(tn, 1e-300)
        clockwise = signed_area(sample_loop(C, 10)) < 0           # interior on the right of the direction of travel
        n_uv = np.stack([tuv[:, 1], -tuv[:, 0]], 1) if clockwise else np.stack([-tuv[:, 1], tuv[:, 0]], 1)
        nrm = np.einsum("mdi,mi->md", J, n_uv)
        nrm = nrm - np.sum(nrm * t, 1, keepdims=True) * t          # in the tangent plane, orthogonal to the curve
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-300)
        vm, w = stress_energy_at(model, U, uvc[:, 0], uvc[:, 1])
        out["uv"][k], out["x"][k], out["normal"][k] = uv, X, nrm
        out["ds"][k], out["vm"][k], out["w"][k], out["inside"][k] = tn[:, 0] / n, vm, w, inside
    return out


# ------------------------------------------------------------------------------------------------ design fields
def _segment_distance(P, Q):
    """Distance from points P (M, 2) to the closed polyline Q (S, 2)."""
    A, B = Q, np.roll(Q, -1, axis=0)
    d = np.full(len(P), np.inf)
    for s in range(0, len(A), 64):
        a, b = A[s:s + 64], B[s:s + 64]
        ab = b - a
        ap = P[:, None, :] - a[None, :, :]
        t = np.clip(np.sum(ap * ab[None], 2) / np.maximum(np.sum(ab * ab, 1)[None], 1e-300), 0.0, 1.0)
        foot = a[None] + t[..., None] * ab[None]
        d = np.minimum(d, np.min(np.linalg.norm(P[:, None, :] - foot, axis=2), 1))
    return d


def passive_sets(prob, host, u, v):
    """Exact passive geometry at parameter points: the void of the bolt holes and polygon cut-outs, and the solid
    of the bolt rings. Every material state is composed as (state & ~void) | solid; evaluate.Evaluator.passive_sets
    delegates here, so that all material tests share one rule."""
    void, solid = np.zeros(len(u), bool), np.zeros(len(u), bool)
    if prob["passive"] and len(u):                           # the host basis cannot be evaluated at no points
        pts = np.stack([u, v], 1)
        X = host.eval(u, v)
        for ps in prob["passive"]:
            if ps["kind"] == "circle":
                d = np.hypot(X[:, 0] - ps["centre"][0], X[:, 1] - ps["centre"][1])
                void |= d < ps["radius"]
                solid |= (d >= ps["radius"]) & (d < ps["radius"] + ps["ring"])
            else:
                void |= Path(np.asarray(ps["uv"])).contains_points(pts)
    return void, solid


def in_passive(prob, host, u, v):
    """Points inside a passive region (hole, ring or cut-out): a loop there bounds no free boundary of the face."""
    void, solid = passive_sets(prob, host, u, v)
    return void | solid


def material_mask(prob, host, u, v, loops_uv):
    """Crisp material test on parameter points, including the passive regions (as in evaluate.loop_material)."""
    pts = np.stack([u, v], 1)
    mat = np.ones(len(u), bool)
    for C in loops_uv:
        mat &= ~Path(sample_loop(np.asarray(C), 20)).contains_points(pts)
    void, solid = passive_sets(prob, host, u, v)
    return (mat & ~void) | solid


def design_field(prob, loops_uv, n=GRID):
    """(2, n, n) float32: crisp indicator (+1 material, -1 void) at the cell centres of the conditioning grid, and
    the signed distance to the nearest trimming curve or passive boundary, measured in the scaled coordinates
    z = (Lu u, Lv v) of the host (physical lengths to first order), divided by sqrt(area), positive in material."""
    host = Host.from_dict(prob["host"]).build(8, 8)
    c = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(c, c, indexing="ij")
    u, v = U.ravel(), V.ravel()
    L = np.array([host.Lu, host.Lv])
    mat = material_mask(prob, host, u, v, loops_uv)
    P = np.stack([u, v], 1) * L
    d = np.full(len(u), np.inf)
    for C in loops_uv:
        d = np.minimum(d, _segment_distance(P, sample_loop(np.asarray(C), 20) * L))
    for ps in prob["passive"]:
        if ps["kind"] == "circle":                               # rectangle host: z equals the physical point
            r = np.hypot(P[:, 0] - ps["centre"][0], P[:, 1] - ps["centre"][1])
            d = np.minimum(d, np.abs(r - ps["radius"]))
        else:
            d = np.minimum(d, _segment_distance(P, np.asarray(ps["uv"], float) * L))
    d = np.where(np.isfinite(d), d, np.sqrt(host.area))
    sdf = np.where(mat, d, -d) / np.sqrt(host.area)
    return np.stack([np.where(mat, 1.0, -1.0).reshape(n, n), sdf.reshape(n, n)]).astype(np.float32)
