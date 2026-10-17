"""Spline host patches: exact coarse control nets for every host family, knot refinement and geometric queries.

All hosts are biquadratic NURBS patches on the parameter square. Control-point index = iu * ncv + iv.
Planar families (rect-ell) map to R^2, shell families (cyl-hood) to R^3.
"""
import math

import numpy as np

from .basis import gauss_grid, greville, nurbs2d

PLANAR = ("rect", "sector", "plate", "arc", "taper", "bolted", "freeform", "ell")
SHELL = ("cyl", "saddle", "dome", "torus", "hood")
FAMILY_NAMES = {
    "rect": "rectangle", "sector": "annular sector", "plate": "plate with a hole", "arc": "curved beam",
    "taper": "tapered patch", "bolted": "rectangle with bolt holes", "freeform": "freeform patch", "ell": "L-shaped host",
    "cyl": "cylindrical panel", "saddle": "saddle", "dome": "dome", "torus": "torus sector", "hood": "freeform hood",
}


# ------------------------------------------------------------------------------------------------ knot insertion
def insert_knot(U, Pw, t, axis):
    """Boehm insertion of one knot t into knot vector U along axis (0: u, 1: v) of the homogeneous net Pw."""
    p = 2
    U = np.asarray(U, float)
    k = int(np.searchsorted(U, t, side="right") - 1)
    Q = np.moveaxis(Pw, axis, 0)
    n = Q.shape[0]
    new = np.empty((n + 1,) + Q.shape[1:])
    new[: k - p + 1] = Q[: k - p + 1]
    for i in range(k - p + 1, k + 1):
        a = (t - U[i]) / (U[i + p] - U[i])
        new[i] = a * Q[i] + (1 - a) * Q[i - 1]
    new[k + 1:] = Q[k:]
    return np.insert(U, k + 1, t), np.moveaxis(new, 0, axis)


def refine(Uu, Uv, Pw, nelu, nelv):
    """Uniform h-refinement to nelu x nelv elements; existing knots are kept (single insertion keeps C1)."""
    for axis, (U, n) in enumerate(((Uu, nelu), (Uv, nelv))):
        have = {round(x, 12) for x in U}
        Uc = np.asarray(U, float)
        for k in range(1, n):
            t = k / n
            if round(t, 12) in have:
                continue
            Uc, Pw = insert_knot(Uc, Pw, t, axis)
        if axis == 0:
            Uu = Uc
        else:
            Uv = Uc
    return Uu, Uv, Pw


def arc3(angle, radius=1.0, start=0.0):
    """Rational quadratic arc (single span, angle < pi): 3 control points (2D) and weights."""
    h = angle / 2
    a0, a2 = start, start + angle
    P0 = radius * np.array([math.cos(a0), math.sin(a0)])
    P2 = radius * np.array([math.cos(a2), math.sin(a2)])
    P1 = radius / math.cos(h) * np.array([math.cos(start + h), math.sin(start + h)])
    return np.array([P0, P1, P2]), np.array([1.0, math.cos(h), 1.0])


# ------------------------------------------------------------------------------------------------ Host
class Host:
    """Biquadratic NURBS host. `spec` holds the coarse exact net; `build(nelu, nelv)` refines it."""

    def __init__(self, family, params, Uu, Uv, P, W, knot_div=(1, 1)):
        self.family, self.params = family, dict(params)
        self.Uu0, self.Uv0 = np.asarray(Uu, float), np.asarray(Uv, float)
        self.P0, self.W0 = np.asarray(P, float), np.asarray(W, float)
        self.dim = self.P0.shape[-1]
        self.knot_div = knot_div
        self.build(2, 2)
        self.Lu, self.Lv, self.area = self._lengths()

    # -- construction ---------------------------------------------------------------------------------------------
    def build(self, nelu, nelv):
        du, dv = self.knot_div
        nelu = max(du, int(round(nelu / du)) * du)
        nelv = max(dv, int(round(nelv / dv)) * dv)
        Pw = np.concatenate([self.P0 * self.W0[..., None], self.W0[..., None]], -1)
        Uu, Uv, Pw = refine(self.Uu0, self.Uv0, Pw, nelu, nelv)
        self.Uu, self.Uv = Uu, Uv
        self.W = Pw[..., -1]
        self.P = (Pw[..., :-1] / self.W[..., None]).reshape(-1, self.dim)
        self.ncu, self.ncv = self.W.shape
        self.nelu, self.nelv = len(np.unique(Uu)) - 1, len(np.unique(Uv)) - 1
        return self

    def mesh_for(self, n_elements):
        """Element counts with about n_elements elements and elements of roughly unit aspect ratio."""
        r = self.Lu / self.Lv
        nelu = max(4, int(round(math.sqrt(n_elements * r))))
        nelv = max(4, int(round(n_elements / nelu)))
        return nelu, nelv

    # -- evaluation -----------------------------------------------------------------------------------------------
    def basis(self, u, v, nd=1):
        return nurbs2d(self.Uu, self.Uv, self.W, np.clip(u, 0, 1), np.clip(v, 0, 1), nd=nd)

    def eval(self, u, v):
        b = self.basis(u, v, nd=0)
        return np.einsum("mk,mkd->md", b["R"], self.P[b["ien"]])

    def jac(self, u, v):
        """Points X (m, dim) and Jacobian J (m, dim, 2)."""
        b = self.basis(u, v, nd=1)
        Pe = self.P[b["ien"]]
        X = np.einsum("mk,mkd->md", b["R"], Pe)
        J = np.stack([np.einsum("mk,mkd->md", b["Ru"], Pe), np.einsum("mk,mkd->md", b["Rv"], Pe)], -1)
        return X, J

    def metric(self, u, v):
        _, J = self.jac(u, v)
        return np.einsum("mdi,mdj->mij", J, J)

    def area_element(self, u, v):
        return np.sqrt(np.maximum(np.linalg.det(self.metric(u, v)), 0.0))

    def curvature(self, u, v):
        """Second fundamental form b_ab of a shell host (m, 2, 2); zeros for planar hosts."""
        m = np.size(u)
        if self.dim == 2:
            return np.zeros((m, 2, 2))
        b = self.basis(u, v, nd=2)
        Pe = self.P[b["ien"]]
        d = {k: np.einsum("mk,mkd->md", b[k], Pe) for k in ("Ru", "Rv", "Ruu", "Ruv", "Rvv")}
        n = np.cross(d["Ru"], d["Rv"])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        B = np.zeros((m, 2, 2))
        B[:, 0, 0] = np.sum(d["Ruu"] * n, 1)
        B[:, 0, 1] = B[:, 1, 0] = np.sum(d["Ruv"] * n, 1)
        B[:, 1, 1] = np.sum(d["Rvv"] * n, 1)
        return B

    def _lengths(self, n=48):
        s = (np.arange(n) + 0.5) / n
        U, V = np.meshgrid(s, s, indexing="ij")
        _, J = self.jac(U.ravel(), V.ravel())
        Lu = float(np.mean(np.linalg.norm(J[:, :, 0], axis=1)))
        Lv = float(np.mean(np.linalg.norm(J[:, :, 1], axis=1)))
        u, v, w, _, _, _ = gauss_grid(self.Uu, self.Uv, 3)
        area = float(np.sum(self.area_element(u, v) * w))
        return Lu, Lv, area

    def min_jacobian_ratio(self, n=64):
        s = (np.arange(n) + 0.5) / n
        U, V = np.meshgrid(s, s, indexing="ij")
        dA = self.area_element(U.ravel(), V.ravel())
        return float(dA.min() / dA.mean())

    def greville_uv(self):
        return greville(self.Uu), greville(self.Uv)

    # -- serialization ----------------------------------------------------------------------------------------------
    def to_dict(self):
        return {"family": self.family, "params": self.params, "Uu": self.Uu0.tolist(), "Uv": self.Uv0.tolist(),
                "P": self.P0.tolist(), "W": self.W0.tolist(), "knot_div": list(self.knot_div)}

    @staticmethod
    def from_dict(d):
        return Host(d["family"], d["params"], d["Uu"], d["Uv"], d["P"], d["W"], tuple(d["knot_div"]))


BEZ = [0, 0, 0, 1, 1, 1]


def _net(points, weights=None):
    P = np.asarray(points, float)
    W = np.ones(P.shape[:2]) if weights is None else np.asarray(weights, float)
    return P, W


# ------------------------------------------------------------------------------------------------ planar families
def rectangle(L, H):
    P = [[(x, y) for y in (0.0, H / 2, H)] for x in (0.0, L / 2, L)]
    return _net(P)


def make_host(family, rng):
    """Sample a host of the given family. Returns Host (parameters drawn from the family ranges)."""
    if family in ("rect", "bolted", "ell"):
        if family == "ell":
            L = float(rng.uniform(1.0, 1.5))
            H = L
        else:
            H = 1.0
            L = float(rng.uniform(1.0, 4.0) if family == "rect" else rng.uniform(1.5, 3.0))
        P, W = rectangle(L, H)
        return Host(family, {"L": L, "H": H}, BEZ, BEZ, P, W)
    if family in ("sector", "arc"):
        if family == "sector":
            ang = math.radians(rng.uniform(45, 150))
            r_in = 1.0
            r_out = float(rng.uniform(1.5, 3.0))
        else:
            kappa = float(rng.uniform(0.2, 1.0))
            width = float(rng.uniform(0.2, 0.5))
            rc = 1.0 / kappa
            r_in, r_out = rc - width / 2, rc + width / 2
            ang = min(math.radians(150), 3.0 / rc)
        A, w = arc3(ang)
        radii = (r_in, 0.5 * (r_in + r_out), r_out)
        P = [[tuple(r * A[i]) for r in radii] for i in range(3)]
        W = [[w[i]] * 3 for i in range(3)]
        return Host(family, {"angle_deg": math.degrees(ang), "r_in": r_in, "r_out": r_out}, BEZ, BEZ, P, W)
    if family == "plate":
        R = 1.0
        L = float(R / rng.uniform(0.2, 0.5))
        a, w = math.sqrt(2) - 1, 0.5 * (1 + math.sqrt(2) / 2)
        hole = [(R, 0), (R, a * R), (a * R, R), (0, R)]
        outer = [(L, 0), (L, L), (L, L), (0, L)]
        mid = [tuple(0.5 * (np.array(h) + np.array(o))) for h, o in zip(hole, outer)]
        P = [[hole[i], mid[i], outer[i]] for i in range(4)]
        W = [[(1.0, w, w, 1.0)[i], 1.0, 1.0] for i in range(4)]
        return Host(family, {"R_hole": R, "L": L}, [0, 0, 0, 0.5, 1, 1, 1], BEZ, P, W, knot_div=(2, 1))
    if family == "taper":
        L, H = float(rng.uniform(1.5, 3.0)), 1.0
        taper = float(rng.uniform(0.3, 1.0))
        top = lambda x: H * (1 - (1 - taper) * x / L)  # noqa: E731
        off = lambda x: 0.5 * (H - top(x))              # noqa: E731
        P = [[(x, off(x) + t * top(x)) for t in (0.0, 0.5, 1.0)] for x in (0.0, L / 2, L)]
        return Host(family, {"L": L, "H": H, "taper": taper}, BEZ, BEZ, *_net(P))
    if family == "freeform":
        for _ in range(50):
            xs, ts = np.linspace(0.0, 3.0, 5), np.linspace(-1.0, 1.0, 4)
            arch, taper = rng.uniform(0.0, 0.5), rng.uniform(0.0, 0.35)
            P = []
            for i, x in enumerate(xs):
                half, centre = 0.75 - taper * x / 3.0, arch * math.sin(math.pi * x / 3.0)
                row = []
                for j, t in enumerate(ts):
                    dx, dy = rng.normal(0.0, 0.08, 2) if (0 < i < 4 and 0 < j < 3) else (0.0, 0.0)
                    row.append((x + dx, centre + half * t + dy))
                P.append(row)
            h = Host(family, {"arch": float(arch), "taper": float(taper)}, [0, 0, 0, 1 / 3, 2 / 3, 1, 1, 1],
                     [0, 0, 0, 0.5, 1, 1, 1], *_net(P), knot_div=(3, 2))
            if h.min_jacobian_ratio() > 0.2:
                return h
        raise RuntimeError("no valid freeform host")
    return make_shell_host(family, rng)


# ------------------------------------------------------------------------------------------------ shell families
def make_shell_host(family, rng):
    if family == "cyl":
        R, ang, Lc = float(rng.uniform(0.8, 2.0)), math.radians(rng.uniform(45, 150)), float(rng.uniform(1.0, 3.0))
        A, w = arc3(ang, R, start=math.pi / 2 - ang / 2)
        P = [[(A[i, 0], y, A[i, 1]) for y in (0.0, Lc / 2, Lc)] for i in range(3)]
        W = [[w[i]] * 3 for i in range(3)]
        return Host(family, {"R": R, "angle_deg": math.degrees(ang), "length": Lc}, BEZ, BEZ, P, W)
    if family in ("saddle", "dome"):
        a = float(rng.uniform(0.75, 1.5))
        c = float(rng.uniform(0.2, 1.0))
        b = [a * a, -a * a, a * a]
        xs = (-a, 0.0, a)
        if family == "saddle":
            P = [[(xs[i], xs[j], 0.5 * c / a * (b[i] - b[j])) for j in range(3)] for i in range(3)]
        else:
            P = [[(xs[i], xs[j], 0.5 * c / a * (2 * a * a - b[i] - b[j])) for j in range(3)] for i in range(3)]
        return Host(family, {"a": a, "c": c}, BEZ, BEZ, *_net(P))
    if family == "torus":
        Rm, r = float(rng.uniform(1.5, 3.0)), float(rng.uniform(0.4, 0.8))
        au, av = math.radians(rng.uniform(45, 120)), math.radians(rng.uniform(60, 150))
        C, wu = arc3(au)
        D, wv = arc3(av, start=-av / 2)
        P = [[((Rm + r * D[j, 0]) * C[i, 0], (Rm + r * D[j, 0]) * C[i, 1], r * D[j, 1]) for j in range(3)]
             for i in range(3)]
        W = [[wu[i] * wv[j] for j in range(3)] for i in range(3)]
        return Host(family, {"R_major": Rm, "r_minor": r, "angle_u_deg": math.degrees(au),
                             "angle_v_deg": math.degrees(av)}, BEZ, BEZ, P, W)
    if family == "hood":
        for _ in range(50):
            xs, ys = np.linspace(0, 2.0, 6), np.linspace(0, 1.5, 5)
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            Z = 0.25 * np.exp(-((X - rng.uniform(0.6, 1.4)) ** 2 + (Y - 0.75) ** 2) / rng.uniform(0.3, 0.8))
            Z += 0.15 * (X / 2.0) ** 2 * rng.uniform(-1, 1) + rng.normal(0, 0.03, Z.shape)
            X[1:-1, 1:-1] += rng.normal(0, 0.03, (4, 3))
            P = np.stack([X, Y, Z], -1)
            h = Host(family, {"seed_bump": float(Z.max())}, [0, 0, 0, 1 / 4, 1 / 2, 3 / 4, 1, 1, 1],
                     [0, 0, 0, 1 / 3, 2 / 3, 1, 1, 1], *_net(P), knot_div=(4, 3))
            if h.min_jacobian_ratio() > 0.2:
                return h
        raise RuntimeError("no valid hood host")
    raise ValueError(family)
