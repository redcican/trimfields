"""CAD-native refinement R_m: gradient steps on the loop control points plus discrete topology moves."""
import math
import time

import jax.numpy as jnp
import numpy as np
from matplotlib.path import Path
from shapely.geometry import Polygon
from shapely.ops import unary_union

from . import indicator as I
from .loops import fit_periodic, resample_closed, sample_loop
from .problems import covers_keep


class Refiner:
    """Refines a loop set on the model an Indicator was built on; compiled functions are shared module-wide."""

    def __init__(self, model, ind, cfg, keep_geoms=()):
        self.model, self.ind, self.cfg = model, ind, cfg
        self.keep_geoms = list(keep_geoms)            # protected regions (problems.keep_geometry)
        self.w = jnp.asarray([cfg["alpha"], cfg["beta"], cfg["gamma"], cfg["w_min_h"] * ind.h])

    def E(self, Cz, act):
        return self.ind.modulus(Cz, act)

    def vol(self, Cz, act):
        return self.ind.volume(Cz, act)

    def pen(self, Cz, act):
        return I.penalty_vg(jnp.asarray(Cz), jnp.asarray(act), self.ind.c, self.w)

    # -- helpers --------------------------------------------------------------------------------------------------
    def _slots(self, loops_uv):
        k, n = self.ind.k_max, self.ind.n_cp
        Cz = np.zeros((k, n, 2))
        act = np.zeros(k)
        for i, C in enumerate(loops_uv[:k]):
            Cz[i], act[i] = np.asarray(C) * self.ind.L, 1.0
        # inactive slots: tiny loops far outside the domain (no influence on chi)
        for i in range(len(loops_uv), k):
            Cz[i] = -50.0 + 1e-3 * np.stack([np.cos(np.arange(n)), np.sin(np.arange(n))], 1)
        return Cz, act

    def _J_and_grad(self, Cz, act):
        st = self.ind.state(Cz, act)
        E = I.EMIN + st["chi"][: self.ind.m] * (1.0 - I.EMIN)
        J, U = self.model.solve(E)
        ct = np.zeros(self.ind.m_pad)
        ct[: self.ind.m] = -self.model.energy_density(U) * (1.0 - I.EMIN)
        return J, U, self.ind.chi_vjp(st, ct)

    def _loop_area(self, Cz_k):
        poly = sample_loop(Cz_k / self.ind.L, 10)
        inside = Path(poly).contains_points(np.stack([self.model.u, self.model.v], 1))
        return float(self.model.dA[inside].sum())

    # -- topology moves -------------------------------------------------------------------------------------------
    def _topology(self, Cz, act, U, it, maxit, vol, vstar, events):
        el_area = self.model.area / self.model.nel
        n = self.ind.n_cp
        for k in np.flatnonzero(act > 0):                            # deletion
            if self._loop_area(Cz[k]) < 0.5 * el_area:
                act[k] = 0.0
                Cz[k] = -50.0 + 1e-3 * np.stack([np.cos(np.arange(n)), np.sin(np.arange(n))], 1)
                events.append({"it": it, "move": "delete", "slot": int(k)})
        w_min = self.cfg["w_min_h"] * self.ind.h
        ids = list(np.flatnonzero(act > 0))
        polys = {k: Polygon(sample_loop(Cz[k], 10)).buffer(0) for k in ids}
        for a in ids:                                                 # merging of touching loops
            for b in ids:
                if b <= a or act[a] == 0 or act[b] == 0:
                    continue
                if polys[a].distance(polys[b]) < 0.25 * w_min:
                    u = unary_union([polys[a].buffer(0.1 * w_min), polys[b].buffer(0.1 * w_min)])
                    if u.geom_type != "Polygon":
                        continue
                    ring = np.asarray(u.exterior.coords)[:-1]
                    Cz[a] = fit_periodic(resample_closed(ring, 200), n)
                    act[b] = 0.0
                    Cz[b] = -50.0 + 1e-3 * np.stack([np.cos(np.arange(n)), np.sin(np.arange(n))], 1)
                    polys[a] = Polygon(sample_loop(Cz[a], 10)).buffer(0)
                    events.append({"it": it, "move": "merge", "slots": [int(a), int(b)]})
        free = np.flatnonzero(act == 0)
        if (self.cfg["nucleation"] and len(free) and it <= 0.6 * maxit and vol >= vstar - 0.005):
            self._nucleate(Cz, act, U, free[0], it, events)

    def _nucleate(self, Cz, act, U, slot, it, events):
        m = self.model
        E = np.asarray(self.E(Cz, act))[: self.ind.m]
        if m.shell:
            field = m.energy_density(U) / m.dA                        # energy density per unit area
        else:
            s = m.stress(U)
            field = math.pi * (4 * (s[:, 0] ** 2 + s[:, 1] ** 2 + 2 * s[:, 2] ** 2) - (s[:, 0] + s[:, 1]) ** 2)
        uv = np.stack([m.u, m.v], 1)
        ok = (E > 0.99) & np.all((uv > 0.06) & (uv < 0.94), 1)
        zq = uv * self.ind.L
        for k in np.flatnonzero(act > 0):
            poly = sample_loop(Cz[k], 6)
            d = np.min(np.linalg.norm(zq[:, None, :] - poly[None, ::2, :], axis=2), axis=1)
            ok &= d > 3 * self.ind.h
        if ok.sum() < 10:
            return
        ref = np.median(field[E > 0.99])
        i = np.flatnonzero(ok)[np.argmin(field[ok])]
        if field[i] > self.cfg["nucleation_tau"] * ref:
            return
        G = m.host.metric(m.u[i:i + 1], m.v[i:i + 1])[0] / np.outer(self.ind.L, self.ind.L)
        Lc = np.linalg.cholesky(G)                                     # physical length = |L^T dz|
        th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        r0 = self.cfg["nucleation_radius_h"] * self.ind.h
        circ = zq[i] + (np.linalg.solve(Lc.T, r0 * np.stack([np.cos(th), np.sin(th)]))).T
        Cz[slot] = fit_periodic(circ, self.ind.n_cp)
        act[slot] = 1.0
        events.append({"it": it, "move": "nucleate", "slot": int(slot), "uv": [float(m.u[i]), float(m.v[i])]})

    # -- volume projection and validity ------------------------------------------------------------------------------
    def _project_volume(self, Cz, act, vstar, iters=4):
        """Newton steps along the volume gradient until V = vstar (steps capped at 0.5 h per control point)."""
        for _ in range(iters):
            V, gV = self.vol(Cz, act)
            err = float(V) - vstar
            if abs(err) < 2e-4:
                break
            gVn = np.asarray(gV) * act[:, None, None]
            den = float(np.sum(gVn**2))
            if den <= 0:
                break
            step = err / den * gVn
            nrm = np.linalg.norm(step, axis=2, keepdims=True)
            Cz = Cz - step * np.minimum(1.0, 0.5 * self.ind.h / (nrm + 1e-30))
        return Cz

    def _valid(self, Cz, act):
        from shapely.geometry import LineString
        polys = []
        for k in np.flatnonzero(act > 0):
            P = sample_loop(Cz[k] / self.ind.L, 10)
            if not LineString(np.vstack([P, P[:1]])).is_simple:
                return False
            poly = Polygon(P).buffer(0)
            if self.keep_geoms and covers_keep(poly, self.keep_geoms):
                return False
            polys.append(poly)
        return not any(polys[i].intersects(polys[j]) for i in range(len(polys)) for j in range(i + 1, len(polys)))

    # -- main loop ------------------------------------------------------------------------------------------------
    def run(self, loops_uv, vstar, maxit):
        """Projected gradient: steps tangent to the volume constraint, volume restoration, topology moves, and
        return of the best valid iterate (lowest compliance among admissible designs at V = vstar)."""
        cfg = self.cfg
        h = self.ind.h
        Cz, act = self._slots(loops_uv)
        Cz = self._project_volume(Cz, act, vstar)
        J0, U, _ = self._J_and_grad(Cz, act)
        mom = np.zeros_like(Cz)
        vrms, b1, b2 = 0.0, 0.8, 0.95
        hist, events = [], []
        rejected = 0
        best = None
        t0 = time.perf_counter()
        it = 0
        for it in range(1, maxit + 1):
            lr = h * (cfg["lr1"] + (cfg["lr0"] - cfg["lr1"]) * 0.5 * (1 + math.cos(math.pi * (it - 1) / maxit)))
            J, U, gJ = self._J_and_grad(Cz, act)
            V, gV = self.vol(Cz, act)
            V = float(V)
            if (it % 5 == 1) and abs(V - vstar) < 2e-3 and (best is None or J < best[0]) and self._valid(Cz, act):
                best = (J, Cz.copy(), act.copy(), it, V)
            R, gR = self.pen(Cz, act)
            K, gK = self.ind.keep_penalty(Cz, act)
            g = (gJ / J0 + np.asarray(gR) + cfg.get("gamma_keep", 1.0) * gK) * act[:, None, None]
            gVn = np.asarray(gV) * act[:, None, None]
            den = float(np.sum(gVn**2))
            if den > 0:
                g = g - float(np.sum(g * gVn)) / den * gVn
            mom = b1 * mom + (1 - b1) * g
            gn2 = np.sum(g**2, 2)[act > 0]
            if gn2.size:
                vrms = b2 * vrms + (1 - b2) * float(gn2.mean())
            step = lr * (mom / (1 - b1**it)) / (math.sqrt(vrms / (1 - b2**it)) + 1e-30)
            nrm = np.linalg.norm(step, axis=2, keepdims=True)
            Cz = Cz - step * np.minimum(1.0, 2 * lr / (nrm + 1e-30))
            Cz = self._project_volume(Cz, act, vstar)
            hist.append((it, J, V, float(R)))
            if it % cfg["n_top"] == 0 and it < 0.8 * maxit:
                n_ev = len(events)
                before = (Cz.copy(), act.copy())
                self._topology(Cz, act, U, it, maxit, V, vstar, events)
                if len(events) > n_ev:
                    # keep the moves only if the compliance after volume restoration does not rise
                    J_pre = self._J_and_grad(*before)[0]
                    Cz = self._project_volume(Cz, act, vstar)
                    J_new = self._J_and_grad(Cz, act)[0]
                    if J_new > J_pre * (1.0 + cfg.get("move_accept_tol", 0.0)):
                        Cz, act = before
                        rejected += len(events) - n_ev
                        del events[n_ev:]
                    else:
                        mom[:] = 0.0
            if it >= cfg["min_iters"] and it % 10 == 0 and len(hist) > 20:
                Jold = hist[-21][1]
                if abs(J - Jold) / abs(Jold) < cfg["stop_rel"] and abs(V - vstar) < 2e-3:
                    break
        Jf, _, _ = self._J_and_grad(Cz, act)
        Vf = float(self.vol(Cz, act)[0])
        if abs(Vf - vstar) < 2e-3 and (best is None or Jf < best[0]) and self._valid(Cz, act):
            best = (Jf, Cz.copy(), act.copy(), it, Vf)
        if best is None:                                  # no admissible iterate: keep the final state
            best = (Jf, Cz, act, it, Vf)
        Jb, Cb, ab, it_b, Vb = best
        loops = [Cb[k] / self.ind.L for k in np.flatnonzero(ab > 0)]
        return loops, {"iterations": it, "best_iteration": it_b, "J_smooth_final": float(Jb),
                       "V_smooth_final": float(Vb), "J_smooth_init": float(J0), "events": events,
                       "rejected_moves": rejected,
                       "t_s": time.perf_counter() - t0, "history": np.array(hist, float)}
