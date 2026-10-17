"""Isogeometric linear elasticity on a host patch: plane stress (planar hosts) and Kirchhoff-Love shells.

A Model is built on a point set (u, v, weight, element) that is either a tensor Gauss grid (optimization) or an
adaptive cut-cell set (evaluation). The Young's modulus is given per point, which is all the design influences.
"""
import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .basis import gauss_grid

try:  # optional direct solver (Intel MKL PARDISO); TRIMFIELDS_SOLVER = auto | pardiso | superlu
    import pypardiso  # noqa: F401
    HAVE_PARDISO = True
except Exception:  # noqa: BLE001
    HAVE_PARDISO = False
SOLVER = os.environ.get("TRIMFIELDS_SOLVER", "auto").lower()
USE_PARDISO = HAVE_PARDISO and SOLVER in ("auto", "pardiso")

NU = 0.3


def plane_stress_D(nu=NU):
    f = 1.0 / (1 - nu**2)
    return f * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])


class Model:
    def __init__(self, host, u, v, w, elem, thickness=None, nu=NU):
        self.host = host
        self.shell = host.dim == 3
        self.ndpc = 3 if self.shell else 2
        self.thickness = thickness
        self.D0 = plane_stress_D(nu)
        order = np.argsort(elem, kind="stable")
        self.u, self.v, self.w, self.elem = u[order], v[order], w[order], elem[order]
        self.ncp = host.ncu * host.ncv
        self.ndof = self.ncp * self.ndpc
        self._element_maps()
        self._point_data()
        self.F = np.zeros(self.ndof)
        self.fixed = np.zeros(0, int)
        self.set_bc(self.fixed, self.F)

    @staticmethod
    def on_grid(host, nq, thickness=None):
        u, v, w, e, _, _ = gauss_grid(host.Uu, host.Uv, nq)
        m = Model(host, u, v, w, e, thickness)
        m.gpe = nq * nq
        return m

    # -- element connectivity and sparsity pattern --------------------------------------------------------------------
    def _element_maps(self):
        nel = self.host.nelu * self.host.nelv
        self.nel = nel
        first = np.r_[0, np.flatnonzero(np.diff(self.elem)) + 1]
        self.starts = first
        assert len(first) == nel, "every element needs at least one point"
        b = self.host.basis(self.u[first], self.v[first], nd=0)
        self.ien = b["ien"]                                   # (nel, 9)
        k = self.ndpc
        self.edof = (self.ien[:, :, None] * k + np.arange(k)[None, None, :]).reshape(nel, -1)
        nd = self.edof.shape[1]
        iK = np.repeat(self.edof, nd, axis=1).ravel()
        jK = np.tile(self.edof, (1, nd)).ravel()
        key = iK.astype(np.int64) * self.ndof + jK
        ukey, self.inv = np.unique(key, return_inverse=True)
        self.rows, self.cols = ukey // self.ndof, ukey % self.ndof
        self.nnz = len(ukey)
        self.indptr = np.r_[0, np.cumsum(np.bincount(self.rows, minlength=self.ndof))]
        self.upper = self.rows <= self.cols                    # upper triangle (CSR order kept) for PARDISO SPD
        self.uindptr = np.r_[0, np.cumsum(np.bincount(self.rows[self.upper], minlength=self.ndof))]
        self._ps = None

    # -- strain-displacement data at the points ------------------------------------------------------------------------
    def _point_data(self, chunk=40000):
        m = self.u.size
        nd = self.edof.shape[1]
        self.terms = []                                        # list of (B (m,3,nd), factor)
        self.dA = np.empty(m)
        Bs = [np.empty((m, 3, nd)) for _ in range(2 if self.shell else 1)]
        for s in range(0, m, chunk):
            sl = slice(s, s + chunk)
            if self.shell:
                Bm, Bb, dA = self._shell_B(self.u[sl], self.v[sl], self.elem[sl])
                Bs[0][sl], Bs[1][sl] = Bm, Bb
            else:
                B, dA = self._plane_B(self.u[sl], self.v[sl], self.elem[sl])
                Bs[0][sl] = B
            self.dA[sl] = dA * self.w[sl]
        if self.shell:
            d = self.thickness
            self.terms = [(Bs[0], d), (Bs[1], d**3 / 12.0)]
        else:
            self.terms = [(Bs[0], 1.0)]
        self.DB = [np.einsum("ij,mjk->mik", self.D0, B) for B, _ in self.terms]
        self.area = float(self.dA.sum())

    def _basis_on_elements(self, u, v, elem, nd):
        b = self.host.basis(u, v, nd=nd)
        assert np.array_equal(b["ien"], self.ien[elem]), "point outside its element"
        return b

    def _plane_B(self, u, v, elem):
        b = self._basis_on_elements(u, v, elem, 1)
        Pe = self.host.P[b["ien"]]
        xu, yu = np.sum(b["Ru"] * Pe[..., 0], 1), np.sum(b["Ru"] * Pe[..., 1], 1)
        xv, yv = np.sum(b["Rv"] * Pe[..., 0], 1), np.sum(b["Rv"] * Pe[..., 1], 1)
        detJ = xu * yv - xv * yu
        Rx = (yv[:, None] * b["Ru"] - yu[:, None] * b["Rv"]) / detJ[:, None]
        Ry = (-xv[:, None] * b["Ru"] + xu[:, None] * b["Rv"]) / detJ[:, None]
        m, nen = Rx.shape
        B = np.zeros((m, 3, 2 * nen))
        B[:, 0, 0::2], B[:, 1, 1::2] = Rx, Ry
        B[:, 2, 0::2], B[:, 2, 1::2] = Ry, Rx
        return B, np.abs(detJ)

    def _shell_B(self, u, v, elem):
        """Linear Kirchhoff-Love membrane and bending B matrices in a local Cartesian frame (Kiendl et al. 2009)."""
        b = self._basis_on_elements(u, v, elem, 2)
        Pe = self.host.P[b["ien"]]
        g = {k: np.einsum("mk,mkd->md", b[k], Pe) for k in ("Ru", "Rv", "Ruu", "Ruv", "Rvv")}
        a1, a2 = g["Ru"], g["Rv"]
        a3t = np.cross(a1, a2)
        jac = np.linalg.norm(a3t, axis=1)
        a3 = a3t / jac[:, None]
        m, nen = b["R"].shape
        # membrane (covariant Voigt: e11, e22, 2 e12)
        Bm = np.zeros((m, 3, 3 * nen))
        for d in range(3):
            Bm[:, 0, d::3] = b["Ru"] * a1[:, d:d + 1]
            Bm[:, 1, d::3] = b["Rv"] * a2[:, d:d + 1]
            Bm[:, 2, d::3] = b["Rv"] * a1[:, d:d + 1] + b["Ru"] * a2[:, d:d + 1]
        # bending (linearized change of the second fundamental form)
        Bb = np.zeros((m, 3, 3 * nen))
        a2xa3, a3xa1 = np.cross(a2, a3), np.cross(a3, a1)
        for row, (Rab, aab, fac) in enumerate(((b["Ruu"], g["Ruu"], 1.0), (b["Rvv"], g["Rvv"], 1.0),
                                               (b["Ruv"], g["Ruv"], 2.0))):
            c1 = np.cross(a2, aab) / jac[:, None]
            c2 = np.cross(aab, a1) / jac[:, None]
            s = np.sum(a3 * aab, 1)[:, None] / jac[:, None]
            for d in range(3):
                Bb[:, row, d::3] = fac * (Rab * a3[:, d:d + 1]
                                          + b["Ru"] * (c1[:, d:d + 1] - s * a2xa3[:, d:d + 1])
                                          + b["Rv"] * (c2[:, d:d + 1] - s * a3xa1[:, d:d + 1]))
        # transformation of covariant Voigt strains to a local orthonormal frame
        G = np.stack([np.stack([np.sum(a1 * a1, 1), np.sum(a1 * a2, 1)], 1),
                      np.stack([np.sum(a2 * a1, 1), np.sum(a2 * a2, 1)], 1)], 1)
        Gi = np.linalg.inv(G)
        con1 = Gi[:, 0, 0:1] * a1 + Gi[:, 0, 1:2] * a2
        con2 = Gi[:, 1, 0:1] * a1 + Gi[:, 1, 1:2] * a2
        e1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
        e2 = np.cross(a3, e1)
        g11, g12 = np.sum(e1 * con1, 1), np.sum(e1 * con2, 1)
        g21, g22 = np.sum(e2 * con1, 1), np.sum(e2 * con2, 1)
        T = np.stack([np.stack([g11**2, g12**2, g11 * g12], 1),
                      np.stack([g21**2, g22**2, g21 * g22], 1),
                      np.stack([2 * g11 * g21, 2 * g12 * g22, g11 * g22 + g12 * g21], 1)], 1)
        return np.einsum("mij,mjk->mik", T, Bm), np.einsum("mij,mjk->mik", T, Bb), jac

    # -- assembly and solution -----------------------------------------------------------------------------------------
    def element_matrices(self, E):
        c = E * self.dA
        nd = self.edof.shape[1]
        Ke = np.zeros((self.nel, nd, nd))
        if hasattr(self, "gpe"):
            q = self.gpe
            for (B, f), DB in zip(self.terms, self.DB):
                A = (f * c[:, None, None] * DB).reshape(self.nel, q * 3, nd)
                Ke += np.matmul(B.reshape(self.nel, q * 3, nd).transpose(0, 2, 1), A)
        else:
            for (B, f), DB in zip(self.terms, self.DB):
                for s in range(0, len(c), 20000):
                    sl = slice(s, s + 20000)
                    contrib = np.einsum("mik,mil->mkl", B[sl], f * c[sl, None, None] * DB[sl])
                    np.add.at(Ke, self.elem[sl], contrib)
        return Ke

    def _stiffness_data(self, E):
        data = np.bincount(self.inv, weights=self.element_matrices(E).ravel(), minlength=self.nnz)
        data[self.mask_fixed] = 0.0
        data[self.diag_fixed] = 1.0
        return data

    def stiffness(self, E):
        return sp.csr_matrix((self._stiffness_data(E), self.cols, self.indptr), shape=(self.ndof, self.ndof))

    def set_bc(self, fixed_dofs, F):
        self.fixed = np.unique(np.asarray(fixed_dofs, int))
        self.F = np.asarray(F, float).copy()
        self.F[self.fixed] = 0.0
        fx = np.zeros(self.ndof, bool)
        fx[self.fixed] = True
        self.mask_fixed = fx[self.rows] | fx[self.cols]
        self.diag_fixed = np.where(self.mask_fixed & (self.rows == self.cols))[0]

    def solve(self, E):
        data = self._stiffness_data(E)
        U = None
        if USE_PARDISO:                                        # symmetric positive definite: upper triangle only
            import pypardiso
            if self._ps is None:
                self._ps = pypardiso.PyPardisoSolver(mtype=2)
            Ku = sp.csr_matrix((data[self.upper], self.cols[self.upper], self.uindptr), shape=(self.ndof, self.ndof))
            try:
                self._ps.factorize(Ku)
                U = self._ps.solve(Ku, self.F)
            except Exception:  # noqa: BLE001  (e.g. a zero pivot of a singular matrix: fall back to SuperLU)
                U = None
        if U is None:
            K = sp.csr_matrix((data, self.cols, self.indptr), shape=(self.ndof, self.ndof))
            lu = splu(K.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                      options=dict(SymmetricMode=True))
            U = lu.solve(self.F)
        return float(self.F @ U), U

    def __del__(self):
        ps = getattr(self, "_ps", None)
        if ps is not None:
            try:
                ps.free_memory(everything=True)
            except Exception:  # noqa: BLE001
                pass

    def energy_density(self, U):
        """Strain energy per point for unit modulus, including the area weight: dJ/dE_q = -energy_q."""
        ue = U[self.edof][self.elem]                            # (m, nd)
        out = np.zeros(self.u.size)
        for (B, f), DB in zip(self.terms, self.DB):
            eps = np.einsum("mik,mk->mi", B, ue)
            sig = np.einsum("mik,mk->mi", DB, ue)
            out += f * np.sum(eps * sig, 1)
        return out * self.dA

    def stress(self, U):
        """Plane-stress Cauchy stress (unit modulus) at the points: (m, 3) = [s11, s22, s12]."""
        ue = U[self.edof][self.elem]
        return np.einsum("mik,mk->mi", self.DB[0], ue)

    def density_matrix(self):
        """Sparse (m, ncp) matrix mapping control-point values to point values."""
        b = self.host.basis(self.u, self.v, nd=0)
        m = self.u.size
        return sp.csr_matrix((b["R"].ravel(), (np.repeat(np.arange(m), b["R"].shape[1]), b["ien"].ravel())),
                             shape=(m, self.ncp))


# ------------------------------------------------------------------------------------------------ loads and supports
EDGES = {"u0": (0, 0.0), "u1": (0, 1.0), "v0": (1, 0.0), "v1": (1, 1.0)}


def _edge_points(edge, s):
    fixed_axis, val = EDGES[edge]
    return (np.full_like(s, val), s) if fixed_axis == 0 else (s, np.full_like(s, val))


def edge_load(model, edge, s0, s1, direction, magnitude, nsub=64, ngl=4):
    """Consistent nodal forces of a uniform traction on a parametric edge interval (total force = magnitude)."""
    host = model.host
    xg, wg = np.polynomial.legendre.leggauss(ngl)
    a = np.linspace(s0, s1, nsub + 1)
    h = np.diff(a) / 2
    s = ((a[:-1] + h)[:, None] + h[:, None] * xg[None]).ravel()
    ws = (h[:, None] * wg[None]).ravel()
    uu, vv = _edge_points(edge, s)
    b = host.basis(uu, vv, nd=1)
    Rt = b["Rv"] if EDGES[edge][0] == 0 else b["Ru"]
    tangent = np.einsum("mk,mkd->md", Rt, host.P[b["ien"]])
    ds = np.linalg.norm(tangent, axis=1) * ws
    dvec = np.asarray(direction, float)
    dvec = dvec / np.linalg.norm(dvec)
    F = np.zeros(model.ndof)
    k = model.ndpc
    for d in range(k):
        np.add.at(F, k * b["ien"].ravel() + d, (b["R"] * (ds * magnitude / ds.sum() * dvec[d])[:, None]).ravel())
    return F


def surface_load(model, direction, magnitude, centre=None, radius=None, nq=4):
    """Shell surface load: 'normal' or a vector direction, over the whole surface or a parametric disk."""
    host = model.host
    u, v, w, _, _, _ = gauss_grid(host.Uu, host.Uv, nq)
    if centre is not None:
        keep = (u - centre[0]) ** 2 + (v - centre[1]) ** 2 <= radius**2
        u, v, w = u[keep], v[keep], w[keep]
    b = host.basis(u, v, nd=1)
    Pe = host.P[b["ien"]]
    a1 = np.einsum("mk,mkd->md", b["Ru"], Pe)
    a2 = np.einsum("mk,mkd->md", b["Rv"], Pe)
    n = np.cross(a1, a2)
    dA = np.linalg.norm(n, axis=1) * w
    if isinstance(direction, str):
        dirs = n / np.linalg.norm(n, axis=1, keepdims=True)
    else:
        dirs = np.broadcast_to(np.asarray(direction, float) / np.linalg.norm(direction), (u.size, 3))
    F = np.zeros(model.ndof)
    for d in range(3):
        np.add.at(F, 3 * b["ien"].ravel() + d, (b["R"] * (dA * magnitude / dA.sum() * dirs[:, d])[:, None]).ravel())
    return F


def edge_cps(host, edge, s0, s1, row=0):
    """Control points of an edge (row 0) or of the next row inward (row 1) whose Greville abscissa is in [s0, s1]."""
    gu, gv = host.greville_uv()
    axis, val = EDGES[edge]
    if axis == 0:
        iu = row if val == 0.0 else host.ncu - 1 - row
        g = gv
        sel = np.where((g >= s0 - 1e-12) & (g <= s1 + 1e-12))[0]
        if len(sel) < 2:
            sel = np.argsort(np.abs(g - 0.5 * (s0 + s1)))[:2]
        return iu * host.ncv + sel
    iv = row if val == 0.0 else host.ncv - 1 - row
    g = gu
    sel = np.where((g >= s0 - 1e-12) & (g <= s1 + 1e-12))[0]
    if len(sel) < 2:
        sel = np.argsort(np.abs(g - 0.5 * (s0 + s1)))[:2]
    return sel * host.ncv + iv
