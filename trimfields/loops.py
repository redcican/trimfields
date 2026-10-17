"""Trimming loops: extraction from density fields, periodic cubic fitting, validity checks, canonical form.

Loops are stored in parameter coordinates (u, v); fitting works in scaled coordinates z = (Lu u, Lv v), which are
close to physical lengths, so that control points spread evenly along the physical boundary.
"""
import numpy as np
from shapely.geometry import LineString, Polygon, box
from skimage import measure

from .basis import periodic_basis
from .simp import density_at


def sample_loop(C, per_span=20):
    n = len(C)
    return periodic_basis(n, np.arange(n * per_span) / (n * per_span)) @ C


def signed_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def resample_closed(P, m):
    seg = np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    t = np.linspace(0, s[-1], m, endpoint=False)
    Pc = np.vstack([P, P[:1]])
    return np.stack([np.interp(t, s, Pc[:, 0]), np.interp(t, s, Pc[:, 1])], 1)


def fit_periodic(P, n, iters=8, lam=1e-3):
    """Least-squares closed periodic cubic B-spline with n control points, foot-point parameter correction."""
    m = max(8 * n, len(P))
    X = resample_closed(P, m)
    t = np.arange(m) / m
    D = np.eye(n) - np.roll(np.eye(n), 1, axis=1)
    reg = lam * (D.T @ D) * m / n
    for _ in range(iters):
        B = periodic_basis(n, t)
        C = np.linalg.solve(B.T @ B + reg, B.T @ X)
        dense_t = np.arange(40 * n) / (40 * n)
        S = periodic_basis(n, dense_t) @ C
        d = ((X[:, None, :] - S[None, :, :]) ** 2).sum(-1)
        t = dense_t[np.argmin(d, 1)]
    return C


def extract_void_loops(host, rho, level=0.5, spe=6, min_area_frac=2.0):
    """Closed rho = level contours enclosing voids in (u, v). A solid ring around an extended grid closes every
    contour; loops touching the patch boundary therefore protrude beyond the parameter square."""
    nu, nv = spe * host.nelu, spe * host.nelv
    pad = 3
    us = (np.arange(-pad, nu + pad) + 0.5) / nu
    vs = (np.arange(-pad, nv + pad) + 0.5) / nv
    U, V = np.meshgrid(us, vs, indexing="ij")
    Z = density_at(host, rho, U.ravel(), V.ravel()).reshape(U.shape)
    Z = np.pad(Z, 1, constant_values=1.0)
    us = np.r_[us[0] - 1 / nu, us, us[-1] + 1 / nu]
    vs = np.r_[vs[0] - 1 / nv, vs, vs[-1] + 1 / nv]
    loops = []
    for c in measure.find_contours(Z, level):
        if np.linalg.norm(c[0] - c[-1]) > 1e-9:
            continue
        uv = np.stack([np.interp(c[:-1, 0], np.arange(len(us)), us), np.interp(c[:-1, 1], np.arange(len(vs)), vs)], 1)
        if len(uv) < 8 or signed_area(uv) == 0:
            continue
        loops.append(uv)
    # keep void loops (density below the level just inside the curve) that are not nested in another void loop
    el_area = 1.0 / (host.nelu * host.nelv)
    keep = []
    for i, a in enumerate(loops):
        poly = Polygon(a).buffer(0)
        if poly.is_empty:
            continue
        pt = poly.representative_point()
        if density_at(host, rho, np.array([np.clip(pt.x, 0, 1)]), np.array([np.clip(pt.y, 0, 1)]))[0] >= level:
            continue                                     # material island, not a void
        if any(j != i and Polygon(b).buffer(0).contains(pt) and Polygon(b).buffer(0).area > poly.area
               for j, b in enumerate(loops)):
            continue
        if poly.intersection(box(0, 0, 1, 1)).area < min_area_frac * el_area:
            continue
        keep.append(a)
    return keep


def fit_loops(host, loops_uv, n_cp=16):
    L = np.array([host.Lu, host.Lv])
    out = []
    for P in loops_uv:
        C = fit_periodic(P * L, n_cp) / L
        out.append(C)
    return out


def in_square_area(C):
    poly = Polygon(sample_loop(C, 12)).buffer(0)
    return poly.intersection(box(0, 0, 1, 1)).area


def self_intersections(C, per_span=12):
    ls = LineString(np.vstack([sample_loop(C, per_span), sample_loop(C, per_span)[:1]]))
    return 0 if ls.is_simple else 1


def loop_collisions(Cs):
    polys = [Polygon(sample_loop(C, 12)).buffer(0).intersection(box(0, 0, 1, 1)) for C in Cs]
    return sum(1 for i in range(len(polys)) for j in range(i + 1, len(polys)) if polys[i].intersects(polys[j]))


def canonical(Cs, k_max=16):
    """Canonical slot tensor (k_max, 2 n + 1): loops ordered by centroid (u, then v), clockwise, starting at the
    control point of largest u; the last column is the existence score (+1 / -1)."""
    items = []
    for C in Cs:
        C = np.asarray(C, float)
        if signed_area(C) > 0:
            C = C[::-1]
        C = np.roll(C, -int(np.argmax(C[:, 0])), axis=0)
        c = C.mean(0)
        items.append((round(c[0], 3), round(c[1], 3), C))
    items.sort(key=lambda x: (x[0], x[1]))
    n = len(Cs[0]) if len(Cs) else 16
    X = np.zeros((k_max, 2 * n + 1))
    X[:, -1] = -1.0
    for k, (_, _, C) in enumerate(items[:k_max]):
        X[k, :-1] = C.ravel()
        X[k, -1] = 1.0
    return X
