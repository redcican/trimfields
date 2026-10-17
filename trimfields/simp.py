"""Isogeometric SIMP with control-point densities, optimality-criteria update and a control-net filter."""
import time

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def _check_feasible(dvol, vstar, void, solid, margin=0.02):
    """The volume target must lie between the fixed solid volume and the volume of the design domain."""
    v_min, v_max = float(dvol[solid].sum()), float(dvol[~void].sum())
    if not v_min + margin < vstar < v_max - margin:
        raise ValueError(f"infeasible volume target {vstar:.3f} (fixed solid {v_min:.3f}, domain {v_max:.3f})")


def run_simp(model, vstar, rmin, void=None, solid=None, penal=3.0, maxit=200, tol=0.01, move=0.2, emin=1e-9,
             continuation=False, min_iters=0):
    """OC-based SIMP. With continuation, the penalty rises as 1 + 0.1 it up to `penal` (used for shells, whose
    grey designs otherwise keep membrane continuity), and the loop stops only after reaching the final penalty."""
    ncp = model.ncp
    Nmat = model.density_matrix()
    dvol = Nmat.T @ (model.dA / model.area)
    tree = cKDTree(model.host.P)
    Hf = tree.sparse_distance_matrix(tree, rmin, output_type="coo_matrix")
    Hf = sp.coo_matrix((rmin - Hf.data, (Hf.row, Hf.col)), shape=(ncp, ncp)).tocsr() + sp.eye(ncp) * rmin
    Hs = np.asarray(Hf.sum(1)).ravel()
    void = np.zeros(ncp, bool) if void is None else void
    solid = np.zeros(ncp, bool) if solid is None else solid
    _check_feasible(dvol, vstar, void, solid)
    rho = np.full(ncp, vstar)
    rho[void], rho[solid] = 0.0, 1.0
    t0 = time.perf_counter()
    change, it, J = 1.0, 0, np.nan
    p_it = penal
    while it < maxit and (change > tol or it < min_iters or p_it < penal):
        p_it = min(penal, 1.0 + 0.1 * it) if continuation else penal
        rq = Nmat @ rho
        J, U = model.solve(emin + rq**p_it * (1 - emin))
        dJdE = -model.energy_density(U)
        dc = Nmat.T @ (p_it * rq ** (p_it - 1) * (1 - emin) * dJdE)
        dc = (Hf @ (rho * dc)) / (Hs * np.maximum(1e-3, rho))
        l1, l2 = 0.0, 1e9
        while (l2 - l1) / (l1 + l2) > 1e-4 and l2 > 1e-40:
            lm = 0.5 * (l1 + l2)
            Be = np.maximum(-dc, 0) / np.maximum(dvol * lm, 1e-300)
            rnew = np.clip(rho * np.sqrt(np.minimum(Be, 1e300)), np.maximum(0, rho - move), np.minimum(1, rho + move))
            rnew[void], rnew[solid] = 0.0, 1.0
            l1, l2 = (lm, l2) if dvol @ rnew > vstar else (l1, lm)
        change = float(np.abs(rnew - rho).max())
        rho = rnew
        it += 1
    rq = Nmat @ rho
    J, _ = model.solve(emin + rq**penal * (1 - emin))
    return rho, {"J_simp": float(J), "iterations": it, "final_change": change, "vol": float(dvol @ rho),
                 "grey_percent": float(400 * np.mean(rq * (1 - rq))), "t_s": time.perf_counter() - t0}


def _proj(x, beta, eta=0.5):
    a = np.tanh(beta * eta)
    return (a + np.tanh(beta * (x - eta))) / (a + np.tanh(beta * (1 - eta)))


def _dproj(x, beta, eta=0.5):
    a = np.tanh(beta * eta)
    return beta * (1 - np.tanh(beta * (x - eta)) ** 2) / (a + np.tanh(beta * (1 - eta)))


def run_simp_projected(model, vstar, rmin, penal=3.0, maxit=250, tol=0.005, move=0.2, emin=1e-6, beta_max=16.0,
                       beta_every=25, solid=None):
    """SIMP with a density filter and Heaviside projection (beta doubled every `beta_every` iterations up to
    beta_max). Used for shells, where plain SIMP stays grey because bending stiffness scales like thickness cubed.
    Control points in `solid` are held at 1 after projection."""
    ncp = model.ncp
    Nmat = model.density_matrix()
    dvol = Nmat.T @ (model.dA / model.area)
    tree = cKDTree(model.host.P)
    Hf = tree.sparse_distance_matrix(tree, rmin, output_type="coo_matrix")
    Hf = sp.coo_matrix((rmin - Hf.data, (Hf.row, Hf.col)), shape=(ncp, ncp)).tocsr() + sp.eye(ncp) * rmin
    Hs = np.asarray(Hf.sum(1)).ravel()
    solid = np.zeros(ncp, bool) if solid is None else solid

    def phys(x_, b_):
        xp_ = _proj((Hf @ x_) / Hs, b_)
        xp_[solid] = 1.0
        return xp_

    _check_feasible(dvol, vstar, np.zeros(ncp, bool), solid)
    x = np.full(ncp, vstar)
    x[solid] = 1.0
    beta, it, change = 1.0, 0, 1.0
    t0 = time.perf_counter()
    while it < maxit and (change > tol or beta < beta_max):
        xt = (Hf @ x) / Hs
        xp = phys(x, beta)
        rq = Nmat @ xp
        J, U = model.solve(emin + rq**penal * (1 - emin))
        dc_xp = Nmat.T @ (penal * rq ** (penal - 1) * (1 - emin) * (-model.energy_density(U)))
        dp = _dproj(xt, beta)
        dp[solid] = 0.0
        dc = Hf @ (dc_xp * dp / Hs)
        dv = Hf @ (dvol * dp / Hs)
        l1, l2 = 0.0, 1e9
        while (l2 - l1) / (l1 + l2) > 1e-4 and l2 > 1e-40:
            lm = 0.5 * (l1 + l2)
            Be = np.maximum(-dc, 0) / np.maximum(np.maximum(dv, 1e-30) * lm, 1e-300)
            xnew = np.clip(x * np.sqrt(np.minimum(Be, 1e300)), np.maximum(0, x - move), np.minimum(1, x + move))
            xnew[solid] = 1.0
            l1, l2 = (lm, l2) if dvol @ phys(xnew, beta) > vstar else (l1, lm)
        change = float(np.abs(xnew - x).max())
        x = xnew
        it += 1
        if it % beta_every == 0 and beta < beta_max:
            beta, change = 2 * beta, 1.0
    rho = phys(x, beta)
    rq = Nmat @ rho
    J, _ = model.solve(emin + rq**penal * (1 - emin))
    return rho, {"J_simp": float(J), "iterations": it, "final_change": change, "vol": float(dvol @ rho),
                 "grey_percent": float(400 * np.mean(rq * (1 - rq))), "beta": beta, "t_s": time.perf_counter() - t0}


def density_at(host, rho, u, v):
    b = host.basis(np.clip(u, 0, 1), np.clip(v, 0, 1), nd=0)
    return np.sum(b["R"] * rho[b["ien"]], 1)
