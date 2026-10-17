"""Conditioning fields on a 64 x 64 grid of the parameter square (independent of the analysis mesh).

Planar (9 channels): G11, G12, G22, log det J, t_u, t_v, support mask, passive mask, V*.
Shell (13 channels): G11, G12, G22, log sqrt(det G), b11, b12, b22, t_u, t_v, t_n, support mask, passive mask, V*.
Metric channels are divided by the host area and curvature channels by sqrt(area), so a uniform scaling
of the host leaves them unchanged.
"""
import numpy as np
from matplotlib.path import Path

from .hosts import Host
from .mechanics import EDGES

GRID = 64
PLANAR_CHANNELS = ("G11", "G12", "G22", "logdetJ", "t_u", "t_v", "support", "passive", "vstar")
SHELL_CHANNELS = ("G11", "G12", "G22", "logdetG", "b11", "b12", "b22", "t_u", "t_v", "t_n", "support", "passive",
                  "vstar")


def _edge_cells(edge, s0, s1, n=GRID):
    axis, val = EDGES[edge]
    idx = np.arange(n)
    c = (idx + 0.5) / n
    sel = idx[(c >= s0 - 0.5 / n) & (c <= s1 + 0.5 / n)]
    if sel.size == 0:
        sel = np.array([int(np.clip(0.5 * (s0 + s1) * n, 0, n - 1))])
    fixed = 0 if val == 0.0 else n - 1
    return (np.full_like(sel, fixed), sel) if axis == 0 else (sel, np.full_like(sel, fixed))


def conditioning(prob, n=GRID):
    host = Host.from_dict(prob["host"]).build(8, 8)
    c = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(c, c, indexing="ij")
    u, v = U.ravel(), V.ravel()
    X, J = host.jac(u, v)
    G = np.einsum("mdi,mdj->mij", J, J)
    A = host.area
    shell = host.dim == 3
    ch = {"G11": G[:, 0, 0] / A, "G12": G[:, 0, 1] / A, "G22": G[:, 1, 1] / A}
    detG = np.linalg.det(G)
    if shell:
        ch["logdetG"] = 0.5 * np.log(detG / A**2)
        B = host.curvature(u, v) / np.sqrt(A)                         # b_ab has units of length
        ch["b11"], ch["b12"], ch["b22"] = B[:, 0, 0], B[:, 0, 1], B[:, 1, 1]
    else:
        ch["logdetJ"] = 0.5 * np.log(detG / A**2)
    shape = (n, n)
    t = np.zeros((n, n, 3))
    sup = np.zeros(shape)
    pas = np.zeros(shape)
    for ld in prob["loads"]:
        if ld["kind"] == "edge":
            iu, iv = _edge_cells(ld["edge"], ld["s0"], ld["s1"], n)
            k = iu * n + iv
            d = np.asarray(ld["dir"], float)
            tp = np.linalg.solve(J[k][:, :2, :], d[None, :, None])[..., 0]           # J^{-1} t (planar)
            tp /= np.linalg.norm(tp, axis=1, keepdims=True)
            t[iu, iv, :2] += ld["mag"] * tp
        elif ld["kind"] == "ring":                                          # bolt bearing load (problems.bearing_load)
            ps = prob["passive"][ld["passive"]]
            dx, dy = X[:, 0] - ps["centre"][0], X[:, 1] - ps["centre"][1]
            dd = np.hypot(dx, dy).reshape(shape)
            ring = (dd >= ps["radius"]) & (dd < ps["radius"] + ps["ring"])
            k = np.flatnonzero(ring.ravel())
            if k.size:
                er = np.stack([dx[k], dy[k]], 1) / np.maximum(dd.ravel()[k], 1e-300)[:, None]
                cosw = np.maximum(er @ np.asarray(ld["dir"], float), 0.0)       # cosine law on the loaded half
                tp = np.linalg.solve(J[k], er[..., None])[..., 0]
                tp /= np.linalg.norm(tp, axis=1, keepdims=True)
                t.reshape(-1, 3)[k, :2] += ld["mag"] * cosw[:, None] * tp
        else:                                                               # shell surface load
            mask = np.ones(n * n, bool)
            if ld["centre"] is not None:
                mask = (u - ld["centre"][0]) ** 2 + (v - ld["centre"][1]) ** 2 <= ld["radius"] ** 2
                if not mask.any():
                    mask[np.argmin((u - ld["centre"][0]) ** 2 + (v - ld["centre"][1]) ** 2)] = True
            nrm = np.cross(J[:, :, 0], J[:, :, 1])
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
            f = nrm if isinstance(ld["dir"], str) else np.broadcast_to(np.asarray(ld["dir"], float), (n * n, 3))
            tn = np.sum(f * nrm, 1)
            ft = f - tn[:, None] * nrm
            tt = np.linalg.solve(G, np.einsum("mdi,md->mi", J, ft)[..., None])[..., 0]
            vals = np.stack([tt[:, 0], tt[:, 1], tn], 1) * mask[:, None]
            t.reshape(-1, 3)[:] += ld["mag"] * vals
    for s in prob["supports"]:
        if s["kind"] == "ring":
            ps = prob["passive"][s["passive"]]
            dd = np.hypot(X[:, 0] - ps["centre"][0], X[:, 1] - ps["centre"][1]).reshape(shape)
            sup[(dd >= ps["radius"]) & (dd < ps["radius"] + ps["ring"])] = 1.0
        else:
            iu, iv = _edge_cells(s["edge"], s["s0"], s["s1"], n)
            sup[iu, iv] = {"all": 1.0, "clamped": 1.0, "pinned": 0.5, "normal": 0.5}[s["dofs"]]
    for ps in prob["passive"]:
        if ps["kind"] == "circle":
            dd = np.hypot(X[:, 0] - ps["centre"][0], X[:, 1] - ps["centre"][1]).reshape(shape)
            pas[dd < ps["radius"]] = -1.0
            pas[(dd >= ps["radius"]) & (dd < ps["radius"] + ps["ring"])] = 1.0
        else:
            pas.ravel()[Path(np.asarray(ps["uv"])).contains_points(np.stack([u, v], 1))] = -1.0
    ch["t_u"], ch["t_v"] = t[..., 0].ravel(), t[..., 1].ravel()
    if shell:
        ch["t_n"] = t[..., 2].ravel()
    ch["support"], ch["passive"] = sup.ravel(), pas.ravel()
    ch["vstar"] = np.full(n * n, prob["vstar"])
    names = SHELL_CHANNELS if shell else PLANAR_CHANNELS
    return np.stack([ch[k].reshape(shape) for k in names]).astype(np.float32)
