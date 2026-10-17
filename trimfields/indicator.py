"""Differentiable material indicator of a loop set (JAX, float64), on CPU or GPU.

Shapes are fixed (active slots bucketed to 1, 2, 4, 8 or 16 loops x n_cp control points; quadrature points padded
to a bucket) and all design data enter as arguments, so a few compiled functions serve every design of a worker.
The all-pairs closest-segment search and the inside test run once per design state without gradient
(`closest`); the indicator is then a function of the distance to the selected segment only (`chi_sel`), whose
reverse-mode derivative costs O(points x loops). This gives the exact gradient almost everywhere.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import shapely
from shapely.geometry import Polygon

from .basis import periodic_basis

jax.config.update("jax_enable_x64", True)
EMIN = 1e-6
MAX_CIRCLES = 3
SLOT_BUCKETS = (1, 2, 4, 8, 16)


def heaviside(s, delta):
    x = jnp.clip(s / delta, -1.0, 1.0)
    return 0.5 + 0.75 * x - 0.25 * x**3


def _seg_d2(dx, dy, ex, ey, g11, g12, g22):
    gex, gey = g11 * ex + g12 * ey, g12 * ex + g22 * ey
    t = jnp.clip(-(dx * gex + dy * gey) / (ex * gex + ey * gey + jnp.finfo(dx.dtype).tiny), 0.0, 1.0)
    px, py = dx + t * ex, dy + t * ey
    return g11 * px**2 + 2 * g12 * px * py + g22 * py**2


def _closest(poly, c):
    """Closest segment of a polygon and inside flag for every point (metric distance; no gradient). Runs in float32:
    it only selects a segment and a side, and the distance itself is recomputed in float64 by _dist_sel."""
    f = jnp.float32
    q, g11, g12, g22 = c["q"].astype(f), c["g11"].astype(f)[:, None], c["g12"].astype(f)[:, None], \
        c["g22"].astype(f)[:, None]
    poly = poly.astype(f)
    a, b = poly, jnp.roll(poly, -1, axis=0)
    qx, qy = q[:, 0:1], q[:, 1:2]
    ax, ay, bx, by = a[:, 0][None], a[:, 1][None], b[:, 0][None], b[:, 1][None]
    d2 = _seg_d2(ax - qx, ay - qy, bx - ax, by - ay, g11, g12, g22)
    j = jnp.argmin(d2, axis=1).astype(jnp.int32)
    cond = (ay > qy) != (by > qy)
    xint = ax + (qy - ay) * (bx - ax) / jnp.where(by == ay, f(1.0), by - ay)
    inside = jnp.sum(cond & (qx < xint), axis=1) % 2 == 1
    return j, inside


def _dist_sel(poly, j, inside, c):
    """Signed metric distance to the selected segment (differentiable in the polygon)."""
    q = c["q"]
    M = poly.shape[0]
    A, B = poly[j], poly[(j + 1) % M]
    d2 = _seg_d2(A[:, 0] - q[:, 0], A[:, 1] - q[:, 1], B[:, 0] - A[:, 0], B[:, 1] - A[:, 1],
                 c["g11"], c["g12"], c["g22"])
    dist = jnp.sqrt(d2 + 1e-24)
    return jnp.where(inside, -dist, dist)


def closest(Cz, c):
    polys = jnp.einsum("mn,knd->kmd", c["S"], jax.lax.stop_gradient(Cz))
    return jax.lax.map(lambda p: _closest(p, c), polys)          # sequential over loops: memory O(points x M)


def chi_sel(Cz, act, c, J, INS):
    polys = jnp.einsum("mn,knd->kmd", c["S"], Cz)
    d = jax.vmap(_dist_sel, in_axes=(0, 0, 0, None))(polys, J, INS, c)
    v = (1.0 - heaviside(d, c["delta"])) * act[:, None]
    H = jnp.prod(1.0 - v, axis=0) * c["p_void"]
    return 1.0 - (1.0 - H) * (1.0 - c["p_solid"])


def chi(Cz, act, c):
    J, INS = closest(Cz, c)
    return chi_sel(Cz, act, c, J, INS)


def modulus(Cz, act, c):
    return EMIN + chi(Cz, act, c) * (1.0 - EMIN)


def volume(Cz, act, c):
    return jnp.sum(chi(Cz, act, c) * c["dA"]) / c["area"]


def penalty(Cz, act, c, w):
    """w = (alpha, beta, gamma, w_min) as an array."""
    alpha, beta, gamma, w_min = w[0], w[1], w[2], w[3]
    d1 = jnp.roll(Cz, -1, 1) - Cz
    d2 = jnp.roll(Cz, -1, 1) - 2 * Cz + jnp.roll(Cz, 1, 1)
    Lseg = jnp.sqrt(jnp.sum(d1**2, 2) + 1e-30)
    reg = jnp.sum(d2**2, (1, 2)) / jnp.sum(d1**2, (1, 2)) + jnp.var(Lseg, 1) / jnp.mean(Lseg, 1) ** 2
    cosang = jnp.sum(jnp.roll(d1, 1, 1) * d1, 2) / (jnp.roll(Lseg, 1, 1) * Lseg)
    fold = jnp.sum(jnp.maximum(0.0, math.cos(math.radians(150)) - cosang) ** 2, 1) \
        + jnp.sum(jnp.maximum(0.0, 0.2 - Lseg / jnp.mean(Lseg, 1, keepdims=True)) ** 2, 1)
    P = jnp.einsum("mn,knd->kmd", c["S"][::2], Cz)
    k = Cz.shape[0]
    diff = P[:, None, :, None, :] - P[None, :, None, :, :]
    dist = jnp.sqrt(jnp.sum(diff**2, -1) + 1e-30)
    pair = act[:, None] * act[None, :] * (1.0 - jnp.eye(k))
    clear = 0.5 * jnp.sum(pair[:, :, None, None] * jnp.maximum(0.0, w_min - dist) ** 2) / w_min**2
    dc = jnp.sqrt(jnp.sum((P[:, :, None, :] - c["circ_c"][None, None]) ** 2, -1))        # (k, M, n_circ)
    circ = jnp.sum(act[:, None, None] * c["circ_on"][None, None]
                   * jnp.maximum(0.0, c["circ_r"][None, None] + 0.5 * w_min - dc) ** 2) / w_min**2
    return jnp.sum(act * (alpha * reg + beta * fold)) + gamma * (clear + circ)


closest_jit = jax.jit(closest)
chi_sel_jit = jax.jit(chi_sel)
chi_sel_vjp = jax.jit(lambda Cz, act, c, J, INS, ct: jax.vjp(lambda C: chi_sel(C, act, c, J, INS), Cz)[1](ct)[0])
penalty_vg = jax.jit(jax.value_and_grad(penalty))


def keep_penalty(Cz, act, ck, margin):
    """Loops must stay at least `margin` away from the sample points of the non-design set (signed metric distance,
    so the penalty also pulls back loops that already cover such points deeply)."""
    polys = jnp.einsum("mn,knd->kmd", ck["S"], Cz)
    J, INS = jax.lax.map(lambda p: _closest(p, ck), jax.lax.stop_gradient(polys))
    d = jax.vmap(_dist_sel, in_axes=(0, 0, 0, None))(polys, J, INS, ck)
    return jnp.sum(act[:, None] * jnp.maximum(0.0, margin - d) ** 2) / margin**2


keep_penalty_vg = jax.jit(jax.value_and_grad(keep_penalty))


class Indicator:
    """Builds the constant arrays of one design (points, metric, passive regions) for the jitted functions."""

    def __init__(self, model, prob, n_cp=16, k_max=16, delta_factor=1.25, samples_per_cp=5, bucket=8192,
                 keep_uv=None):
        host = model.host
        self.L = np.array([host.Lu, host.Lv])
        self.m = model.u.size
        self.m_pad = int(math.ceil(self.m / bucket) * bucket)
        G = host.metric(model.u, model.v) / (self.L[None, :, None] * self.L[None, None, :])
        q = np.full((self.m_pad, 2), -1e3)
        q[: self.m] = np.stack([model.u, model.v], 1) * self.L
        g = np.tile(np.eye(2), (self.m_pad, 1, 1))
        g[: self.m] = G
        dA = np.zeros(self.m_pad)
        dA[: self.m] = model.dA
        nq = int(round(math.sqrt(getattr(model, "gpe", 16))))
        self.h = math.sqrt(model.area / model.nel)
        self.delta = delta_factor * self.h / nq
        self.n_cp, self.k_max = n_cp, k_max
        self.dA_np, self.area, self._cache = dA, float(model.area), None
        pv, ps = self._passive(model, prob)
        circ_c, circ_r, circ_on = np.zeros((MAX_CIRCLES, 2)), np.zeros(MAX_CIRCLES), np.zeros(MAX_CIRCLES)
        for i, p in enumerate([p for p in prob["passive"] if p["kind"] == "circle"][:MAX_CIRCLES]):
            circ_c[i], circ_r[i], circ_on[i] = p["centre"], p["radius"] + p["ring"], 1.0
        S = periodic_basis(n_cp, np.arange(samples_per_cp * n_cp) / (samples_per_cp * n_cp))
        self.ck = self._keep_constants(host, keep_uv, S) if keep_uv is not None and len(keep_uv) else None
        self.keep_margin = 0.5 * self.h
        self.c = {k: jnp.asarray(v) for k, v in dict(
            q=q, g11=g[:, 0, 0], g12=g[:, 0, 1], g22=g[:, 1, 1], dA=dA, area=np.float64(model.area),
            delta=np.float64(self.delta), p_void=pv, p_solid=ps, S=S, circ_c=circ_c, circ_r=circ_r,
            circ_on=circ_on).items()}

    def _keep_constants(self, host, keep_uv, S, bucket=256):
        n = len(keep_uv)
        n_pad = int(math.ceil(n / bucket) * bucket)
        q = np.full((n_pad, 2), -1e3)
        q[:n] = np.asarray(keep_uv) * self.L
        g = np.tile(np.eye(2), (n_pad, 1, 1))
        g[:n] = host.metric(np.clip(keep_uv[:, 0], 0, 1), np.clip(keep_uv[:, 1], 0, 1)) / (
            self.L[None, :, None] * self.L[None, None, :])
        return {k: jnp.asarray(v) for k, v in dict(q=q, g11=g[:, 0, 0], g12=g[:, 0, 1], g22=g[:, 1, 1], S=S).items()}

    def keep_penalty(self, Cz, act):
        """Value and gradient of the keep-region penalty (zero without a non-design set)."""
        if self.ck is None:
            return 0.0, np.zeros_like(np.asarray(Cz))
        v, g = keep_penalty_vg(jnp.asarray(Cz), jnp.asarray(act), self.ck, self.keep_margin)
        return float(v), np.asarray(g)

    def _passive(self, model, prob):
        pv, ps = np.ones(self.m_pad), np.zeros(self.m_pad)
        if not prob["passive"]:
            return pv, ps
        X = model.host.eval(model.u, model.v)
        uvL = np.stack([model.u, model.v], 1) * self.L
        for p in prob["passive"]:
            if p["kind"] == "circle":
                d = np.hypot(X[:, 0] - p["centre"][0], X[:, 1] - p["centre"][1])
                pv[: self.m] *= np.asarray(heaviside(d - p["radius"], self.delta))
                ps[: self.m] = np.maximum(ps[: self.m], ((d >= p["radius"]) & (d < p["radius"] + p["ring"])) * 1.0)
            else:
                poly = Polygon(np.asarray(p["uv"]) * self.L)
                s = shapely.distance(poly.exterior, shapely.points(uvL))
                inside = shapely.contains_xy(poly, uvL[:, 0], uvL[:, 1])
                pv[: self.m] *= np.asarray(heaviside(np.where(inside, -s, s), self.delta))
        return pv, ps

    # -- evaluation with active-slot bucketing and a one-state cache ---------------------------------------------
    def state(self, Cz, act):
        """Indicator of a loop set, with the data needed for gradients. Only active slots are evaluated; they are
        padded with parked loops to the next bucket size so that compiled functions are reused."""
        Cz, act = np.asarray(Cz, float), np.asarray(act, float)
        key = hash((Cz.tobytes(), act.tobytes()))
        if self._cache is not None and self._cache["key"] == key:
            return self._cache
        idx = np.flatnonzero(act > 0)
        kb = next(b for b in SLOT_BUCKETS if b >= max(1, len(idx))) if len(idx) <= SLOT_BUCKETS[-1] else len(idx)
        Czb = np.empty((kb, Cz.shape[1], 2))
        Czb[:] = -50.0 + 1e-3 * np.stack([np.cos(np.arange(Cz.shape[1])), np.sin(np.arange(Cz.shape[1]))], 1)
        Czb[: len(idx)] = Cz[idx]
        actb = np.zeros(kb)
        actb[: len(idx)] = 1.0
        Czb, actb = jnp.asarray(Czb), jnp.asarray(actb)
        J, INS = closest_jit(Czb, self.c)
        chi_ = chi_sel_jit(Czb, actb, self.c, J, INS)
        self._cache = {"key": key, "idx": idx, "Czb": Czb, "actb": actb, "J": J, "INS": INS,
                       "chi": np.asarray(chi_), "shape": Cz.shape}
        return self._cache

    def chi_vjp(self, st, ct):
        """Cotangent ct of chi (length m_pad) -> gradient with respect to all k_max x n_cp x 2 loop coordinates."""
        g = np.asarray(chi_sel_vjp(st["Czb"], st["actb"], self.c, st["J"], st["INS"], jnp.asarray(ct)))
        out = np.zeros(st["shape"])
        out[st["idx"]] = g[: len(st["idx"])]
        return out

    def chi(self, Cz, act):
        return self.state(Cz, act)["chi"]

    def modulus(self, Cz, act):
        return EMIN + self.chi(Cz, act) * (1.0 - EMIN)

    def volume(self, Cz, act):
        """Volume fraction and its gradient."""
        st = self.state(Cz, act)
        V = float(np.sum(st["chi"] * self.dA_np) / self.area)
        return V, self.chi_vjp(st, self.dA_np / self.area)
