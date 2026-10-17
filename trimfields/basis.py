"""Vectorized B-spline / NURBS basis functions with derivatives up to second order (Piegl & Tiller A2.3)."""
import numpy as np


def _sdiv(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def find_span(U, p, u):
    U = np.asarray(U, float)
    n = len(U) - p - 2
    return np.clip(np.searchsorted(U, u, side="right") - 1, p, n)


def ders_basis(U, p, u, nd=1):
    """Non-zero basis functions and derivatives. Returns span (m,), D (nd+1, m, p+1)."""
    U = np.asarray(U, float)
    u = np.atleast_1d(np.asarray(u, float))
    m = u.size
    span = find_span(U, p, u)
    ndu = np.zeros((p + 1, p + 1, m))
    ndu[0, 0] = 1.0
    left = np.zeros((p + 1, m))
    right = np.zeros((p + 1, m))
    for j in range(1, p + 1):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = np.zeros(m)
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = _sdiv(ndu[r, j - 1], ndu[j, r])
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved
    D = np.zeros((nd + 1, p + 1, m))
    for j in range(p + 1):
        D[0, j] = ndu[j, p]
    for r in range(p + 1):
        a = np.zeros((2, p + 1, m))
        a[0, 0] = 1.0
        s1, s2 = 0, 1
        for k in range(1, nd + 1):
            d = np.zeros(m)
            rk, pk = r - k, p - k
            if r >= k:
                a[s2, 0] = _sdiv(a[s1, 0], ndu[pk + 1, rk])
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if r - 1 <= pk else p - r
            for j in range(j1, j2 + 1):
                a[s2, j] = _sdiv(a[s1, j] - a[s1, j - 1], ndu[pk + 1, rk + j])
                d = d + a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = _sdiv(-a[s1, k - 1], ndu[pk + 1, r])
                d = d + a[s2, k] * ndu[r, pk]
            D[k, r] = d
            s1, s2 = s2, s1
    fac = p
    for k in range(1, nd + 1):
        D[k] *= fac
        fac *= p - k
    return span, np.transpose(D, (0, 2, 1))


def nurbs2d(Uu, Uv, W, u, v, nd=1, p=2):
    """Rational tensor-product basis at paired points (u, v). Returns dict with ien (m, nen) and R, Ru, Rv
    (and Ruu, Ruv, Rvv if nd == 2). Control-point index = iu * ncv + iv."""
    u = np.atleast_1d(np.asarray(u, float))
    v = np.atleast_1d(np.asarray(v, float))
    m = u.size
    su, Du = ders_basis(Uu, p, u, max(nd, 1))
    sv, Dv = ders_basis(Uv, p, v, max(nd, 1))
    ncv = W.shape[1]
    iu = su[:, None] - p + np.arange(p + 1)[None, :]
    iv = sv[:, None] - p + np.arange(p + 1)[None, :]
    ien = (iu[:, :, None] * ncv + iv[:, None, :]).reshape(m, -1)
    w = W.ravel()[ien]

    def tp(a, b):
        return (Du[a][:, :, None] * Dv[b][:, None, :]).reshape(m, -1)

    wN, wNu, wNv = w * tp(0, 0), w * tp(1, 0), w * tp(0, 1)
    Wt, Wu, Wv = wN.sum(1)[:, None], wNu.sum(1)[:, None], wNv.sum(1)[:, None]
    R = wN / Wt
    Ru = (wNu - R * Wu) / Wt
    Rv = (wNv - R * Wv) / Wt
    out = {"ien": ien, "R": R, "Ru": Ru, "Rv": Rv}
    if nd >= 2:
        wNuu, wNuv, wNvv = w * tp(2, 0), w * tp(1, 1), w * tp(0, 2)
        Wuu, Wuv, Wvv = wNuu.sum(1)[:, None], wNuv.sum(1)[:, None], wNvv.sum(1)[:, None]
        out["Ruu"] = (wNuu - 2 * Ru * Wu - R * Wuu) / Wt
        out["Ruv"] = (wNuv - Ru * Wv - Rv * Wu - R * Wuv) / Wt
        out["Rvv"] = (wNvv - 2 * Rv * Wv - R * Wvv) / Wt
    return out


def gauss_grid(Uu, Uv, nq):
    """Tensor Gauss points on every knot span. Points of one element are contiguous (element e = i*nelv + j).
    Returns u, v, weight (parametric), element index, nelu, nelv."""
    ku, kv = np.unique(np.asarray(Uu, float)), np.unique(np.asarray(Uv, float))
    x, w = np.polynomial.legendre.leggauss(nq)
    du, dv = np.diff(ku), np.diff(kv)
    mu = 0.5 * (ku[1:] + ku[:-1])
    mv = 0.5 * (kv[1:] + kv[:-1])
    nelu, nelv = len(du), len(dv)
    uu = mu[:, None] + 0.5 * du[:, None] * x[None, :]               # (nelu, nq)
    vv = mv[:, None] + 0.5 * dv[:, None] * x[None, :]               # (nelv, nq)
    wu = 0.5 * du[:, None] * w[None, :]
    wv = 0.5 * dv[:, None] * w[None, :]
    U = np.broadcast_to(uu[:, None, :, None], (nelu, nelv, nq, nq))
    V = np.broadcast_to(vv[None, :, None, :], (nelu, nelv, nq, nq))
    Wq = wu[:, None, :, None] * wv[None, :, None, :]
    e = np.broadcast_to(np.arange(nelu * nelv).reshape(nelu, nelv)[:, :, None, None], (nelu, nelv, nq, nq))
    return U.ravel(), V.ravel(), Wq.ravel(), e.ravel(), nelu, nelv


def greville(U, p=2):
    U = np.asarray(U, float)
    n = len(U) - p - 1
    return np.array([U[i + 1:i + p + 1].mean() for i in range(n)])


def periodic_basis(n, t, der=0):
    """(M, n) matrix: c(t) = B @ C for a uniform periodic cubic B-spline with n control points, t in [0, 1)."""
    t = np.asarray(t, float) % 1.0
    x = t * n
    j = np.minimum(np.floor(x).astype(int), n - 1)
    s = x - j
    if der == 0:
        b = [(1 - s) ** 3 / 6, (3 * s**3 - 6 * s**2 + 4) / 6, (-3 * s**3 + 3 * s**2 + 3 * s + 1) / 6, s**3 / 6]
    else:
        b = [-n * (1 - s) ** 2 / 2, n * (9 * s**2 - 12 * s) / 6, n * (-9 * s**2 + 6 * s + 3) / 6, n * s**2 / 2]
    B = np.zeros((len(t), n))
    rows = np.arange(len(t))
    for k in range(4):
        np.add.at(B, (rows, (j + k) % n), b[k])
    return B
