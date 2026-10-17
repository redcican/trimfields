"""Crisp evaluation with adaptive quadrature on cut elements (finite-cell style), used for every reported number.

Uncut elements: 3 x 3 Gauss points with the element's material state. Elements crossed by the boundary:
4 x 4 sub-cells with 2 x 2 Gauss points each and the crisp material state at every point.
"""
import numpy as np
from matplotlib.path import Path

from .hosts import Host
from .loops import sample_loop
from .mechanics import Model
from .problems import apply_bc
from .simp import density_at

EMIN = 1e-6


def _element_points(Uu, Uv, nsub, nq):
    """Points of every element on an nsub x nsub sub-cell grid with nq x nq Gauss points per sub-cell."""
    ku, kv = np.unique(Uu), np.unique(Uv)
    x, w = np.polynomial.legendre.leggauss(nq)
    loc = (np.arange(nsub)[:, None] + 0.5 * (x[None, :] + 1)) / nsub          # (nsub, nq) in [0, 1]
    wl = (np.ones(nsub)[:, None] * 0.5 * w[None, :]) / nsub
    lu, lv = loc.ravel(), loc.ravel()
    LU, LV = np.meshgrid(lu, lv, indexing="ij")
    WW = np.outer(wl.ravel(), wl.ravel())
    nelu, nelv = len(ku) - 1, len(kv) - 1
    du, dv = np.diff(ku), np.diff(kv)
    u = ku[:-1, None, None, None] + du[:, None, None, None] * LU[None, None]
    v = kv[None, :-1, None, None] + dv[None, :, None, None] * LV[None, None]
    wq = (du[:, None, None, None] * dv[None, :, None, None]) * WW[None, None]
    full = (nelu, nelv) + LU.shape
    e = np.broadcast_to(np.arange(nelu * nelv).reshape(nelu, nelv)[:, :, None, None], full)
    shape = (nelu * nelv, -1)
    return (np.broadcast_to(u, full).reshape(shape), np.broadcast_to(v, full).reshape(shape),
            np.broadcast_to(wq, full).reshape(shape), e.reshape(shape))


class Evaluator:
    def __init__(self, prob, nel, thickness=None, factor=2):
        self.prob = prob
        self.host = Host.from_dict(prob["host"]).build(factor * nel[0], factor * nel[1])
        self.thickness = thickness
        self.base = _element_points(self.host.Uu, self.host.Uv, 1, 3)
        self.fine = _element_points(self.host.Uu, self.host.Uv, 4, 2)
        self.probe = _element_points(self.host.Uu, self.host.Uv, 5, 1)          # 5 x 5 classification samples
        self._fine_dA = None

    # -- material tests -------------------------------------------------------------------------------------------
    def passive_sets(self, u, v):
        """Exact passive geometry at the points: the void of the bolt holes and polygon cut-outs, and the solid of
        the bolt rings (fields.passive_sets, the one rule shared by every material test). Every material state
        becomes (state & ~void) | solid, for loop sets and for thresholded densities alike, so that a bearing load
        always acts on material."""
        from .fields import passive_sets
        return passive_sets(self.prob, self.host, u, v)

    def loop_material(self, loops_uv):
        polys = [Path(sample_loop(np.asarray(C), 20)) for C in loops_uv]

        def fn(u, v):
            pts = np.stack([u, v], 1)
            mat = np.ones(len(u), bool)
            for p in polys:
                mat &= ~p.contains_points(pts)
            void, solid = self.passive_sets(u, v)
            return (mat & ~void) | solid
        return fn

    def density_material(self, rho, th):
        host_opt = self.host_opt                      # bound at creation (threshold_for_volume sets it)

        def fn(u, v):
            void, solid = self.passive_sets(u, v)
            return ((density_at(host_opt, rho, u, v) >= th) & ~void) | solid
        return fn

    # -- evaluation -----------------------------------------------------------------------------------------------
    def evaluate(self, fn, keep_model=False):
        """Crisp compliance and volume fraction of a material test. With keep_model, the result also holds the
        displacement vector and the Model, from which fields at arbitrary points follow (fields.py)."""
        pu, pv, _, _ = self.probe
        state = fn(pu.ravel(), pv.ravel()).reshape(pu.shape)
        full, empty = state.all(1), (~state).all(1)
        cut = ~(full | empty)
        bu, bv, bw, be = self.base
        fu, fv, fw, fe = self.fine
        u = np.concatenate([bu[~cut].ravel(), fu[cut].ravel()])
        v = np.concatenate([bv[~cut].ravel(), fv[cut].ravel()])
        w = np.concatenate([bw[~cut].ravel(), fw[cut].ravel()])
        e = np.concatenate([be[~cut].ravel(), fe[cut].ravel()])
        # no cut element (an untrimmed host whose passive edges fall between the probe samples): skip the test
        mat_cut = fn(fu[cut].ravel(), fv[cut].ravel()) if cut.any() else np.zeros(0, bool)
        mat = np.concatenate([np.repeat(full[~cut], bu.shape[1]), mat_cut])
        model = Model(self.host, u, v, w, e, self.thickness)
        order = np.argsort(e, kind="stable")                  # Model sorts its points by element
        chi = mat[order].astype(float)
        apply_bc(self.prob, model)
        J, U = model.solve(EMIN + chi * (1 - EMIN))
        res = {"J": float(J), "V": float(np.sum(chi * model.dA) / model.area), "cut_elements": int(cut.sum())}
        if keep_model:
            res["U"], res["model"] = U, model
        return res

    def volume(self, fn):
        """Crisp volume fraction of a material test on the fine sub-cell points (no analysis)."""
        fu, fv, fw, _ = self.fine
        if self._fine_dA is None:
            self._fine_dA = self.host.area_element(fu.ravel(), fv.ravel()) * fw.ravel()
        return float(np.sum(self._fine_dA * fn(fu.ravel(), fv.ravel())) / self._fine_dA.sum())

    def threshold_for_volume(self, host_opt, rho, vstar):
        """Density level whose crisp material set has volume fraction vstar (on the fine sub-cell points)."""
        self.host_opt = host_opt
        fu, fv, fw, _ = self.fine
        if self._fine_dA is None:
            self._fine_dA = self.host.area_element(fu.ravel(), fv.ravel()) * fw.ravel()
        r = density_at(host_opt, rho, fu.ravel(), fv.ravel())
        void, solid = self.passive_sets(fu.ravel(), fv.ravel())
        lo, hi = 0.0, 1.0
        for _ in range(40):
            th = 0.5 * (lo + hi)
            V = np.sum(self._fine_dA * (((r >= th) & ~void) | solid)) / self._fine_dA.sum()
            lo, hi = (th, hi) if V > vstar else (lo, th)
        return 0.5 * (lo + hi)
