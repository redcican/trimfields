"""Body-fitted finite-element reference for planar designs (optional; requires gmsh).

The exported STEP face is meshed with quadratic triangles that conform to the trimming curves, and the plane-stress
problem is solved with the continuum boundary conditions of problems.py. This solution does not share the cut-cell
and soft-void errors of the immersed evaluator, so it serves as the reference of the boundary-stress comparison.

    python -m trimfields.bodyfit <design.npz> <design.step> [--size h] [--out reference.npz]

The result holds the mesh, the displacement, the compliance, the volume fraction and the von Mises stress at the
loop samples stored in the design (loop_x), interpolated along the conforming boundary.
"""
import argparse
import json

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .hosts import Host
from .mechanics import EDGES, plane_stress_D

# 6-point Gauss rule on the reference triangle (degree 4); weights sum to 1/2
_A, _B = 0.445948490915965, 0.091576213509771
_WA, _WB = 0.223381589678011 / 2, 0.109951743655322 / 2
GAUSS = np.array([[_A, _A, _WA], [1 - 2 * _A, _A, _WA], [_A, 1 - 2 * _A, _WA],
                  [_B, _B, _WB], [1 - 2 * _B, _B, _WB], [_B, 1 - 2 * _B, _WB]])
G3, W3 = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)]), np.array([5.0, 8.0, 5.0]) / 9.0


def shape_p2(xi, eta):
    """Quadratic triangle (gmsh node order: corners 0-2, mid-sides 3: 0-1, 4: 1-2, 5: 2-0): N (6,), dN (6, 2)."""
    l1, l2, l3 = 1 - xi - eta, xi, eta
    N = np.array([l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), l3 * (2 * l3 - 1), 4 * l1 * l2, 4 * l2 * l3, 4 * l3 * l1])
    dN = np.array([[1 - 4 * l1, 1 - 4 * l1], [4 * l2 - 1, 0.0], [0.0, 4 * l3 - 1],
                   [4 * (l1 - l2), -4 * l2], [4 * l3, 4 * l2], [-4 * l3, 4 * (l1 - l3)]])
    return N, dN


def shape_edge(s):
    """Quadratic edge (nodes: end 0, end 1, middle): N (3,), dN/ds (3,) for s in [-1, 1]."""
    return (np.array([0.5 * s * (s - 1), 0.5 * s * (s + 1), 1 - s * s]),
            np.array([s - 0.5, s + 0.5, -2 * s]))


def mesh_face(step_path, size, order=2):
    """Conforming quadratic triangulation of the STEP face(s) with gmsh: nodes (n, 2), triangles (m, 6)."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5 * size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 24)
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes = coords.reshape(-1, 3)[:, :2]
        renum = {int(t): i for i, t in enumerate(tags)}
        etypes, _, enodes = gmsh.model.mesh.getElements(2)
        tris = []
        for et, en in zip(etypes, enodes):
            if et == 9:                                           # 6-node triangles
                tris.append(np.vectorize(renum.get)(np.asarray(en).reshape(-1, 6)))
        return nodes, np.concatenate(tris)
    finally:
        gmsh.finalize()


def _edge_curve(host, edge, n=800):
    axis, val = EDGES[edge]
    s = np.linspace(0, 1, n)
    u, v = (np.full(n, val), s) if axis == 0 else (s, np.full(n, val))
    return s, host.eval(u, v)


def _nearest_on_curve(P, s, Xc):
    """Nearest point of a polyline (s, Xc) for every point of P: distance and curve parameter."""
    d = np.full(len(P), np.inf)
    sp_ = np.zeros(len(P))
    A, B, sa, sb = Xc[:-1], Xc[1:], s[:-1], s[1:]
    for k in range(0, len(A), 100):
        a, b = A[k:k + 100], B[k:k + 100]
        ab = b - a
        t = np.clip(np.sum((P[:, None, :] - a[None]) * ab[None], 2) / np.maximum(np.sum(ab * ab, 1)[None], 1e-300),
                    0, 1)
        foot = a[None] + t[..., None] * ab[None]
        dist = np.linalg.norm(P[:, None, :] - foot, axis=2)
        j = np.argmin(dist, 1)
        better = dist[np.arange(len(P)), j] < d
        d[better] = dist[np.arange(len(P)), j][better]
        sp_[better] = (sa[k:k + 100][j] + t[np.arange(len(P)), j] * (sb[k:k + 100] - sa[k:k + 100])[j])[better]
    return d, sp_


def boundary_edges(tris):
    """Boundary edges of the quadratic triangulation as (end, end, middle) node triples."""
    edges = {}
    for tri in tris:
        for a, b, m in ((0, 1, 3), (1, 2, 4), (2, 0, 5)):
            key = (min(tri[a], tri[b]), max(tri[a], tri[b]))
            edges[key] = edges.get(key, 0) + 1
            edges.setdefault(("mid",) + key, (tri[a], tri[b], tri[m]))
    return [edges[("mid",) + k] for k, c in edges.items() if k[0] != "mid" and c == 1]


def solve_face(prob, nodes, tris, tol=None):
    """Plane-stress solution on the conforming mesh with the continuum boundary conditions of the problem."""
    host = Host.from_dict(prob["host"]).build(8, 8)
    n = len(nodes)
    diag = np.linalg.norm(nodes.max(0) - nodes.min(0))
    tol = 1e-5 * diag if tol is None else tol
    D = plane_stress_D()
    # assembly
    rows, cols, vals = [], [], []
    area = 0.0
    for tri in tris:
        X = nodes[tri]
        Ke = np.zeros((12, 12))
        for xi, eta, wq in GAUSS:
            N, dN = shape_p2(xi, eta)
            Jm = dN.T @ X                                        # (2, 2): Jm[i, j] = d x_j / d xi_i
            detJ = abs(np.linalg.det(Jm))                        # either orientation of the imported face
            dNx = dN @ np.linalg.inv(Jm).T                       # (6, 2): dN / dx_j = dN / dxi_i * dxi_i / dx_j
            B = np.zeros((3, 12))
            B[0, 0::2], B[1, 1::2] = dNx[:, 0], dNx[:, 1]
            B[2, 0::2], B[2, 1::2] = dNx[:, 1], dNx[:, 0]
            Ke += B.T @ D @ B * detJ * wq
            area += detJ * wq
        dof = np.stack([2 * tri, 2 * tri + 1], 1).ravel()
        rows.append(np.repeat(dof, 12))
        cols.append(np.tile(dof, 12))
        vals.append(Ke.ravel())
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 * n, 2 * n))
    # boundary conditions
    bedges = boundary_edges(tris)
    F = np.zeros(2 * n)
    fixed = set()
    edge_data = {e: _edge_curve(host, e) for e in EDGES}
    for s_ in prob["supports"]:
        if s_["kind"] == "ring":
            ps = prob["passive"][s_["passive"]]
            r = np.hypot(nodes[:, 0] - ps["centre"][0], nodes[:, 1] - ps["centre"][1])
            for i in np.flatnonzero(r <= ps["radius"] + ps["ring"] + tol):
                fixed.update((2 * i, 2 * i + 1))
            continue
        d, sv = _nearest_on_curve(nodes, *edge_data[s_["edge"]])
        on = (d < tol) & (sv >= s_["s0"] - 1e-9) & (sv <= s_["s1"] + 1e-9)
        comps = (int(s_["fixed_component"]),) if s_["dofs"] == "normal" else (0, 1)
        for i in np.flatnonzero(on):
            fixed.update(2 * i + c for c in comps)
    for ld in prob["loads"]:
        Fl = np.zeros(2 * n)
        if ld["kind"] == "edge":
            s, Xc = edge_data[ld["edge"]]
            dvec = np.asarray(ld["dir"], float) / np.linalg.norm(ld["dir"])
            d, sv = _nearest_on_curve(nodes, s, Xc)
            for a, b, m in bedges:                           # mesh edges on the host edge, integrated only over
                if all(d[i] < tol for i in (a, b, m)):       # the part of the segment [s0, s1] (Gauss points)
                    se = sv[[a, b, m]]

                    def uniform(x, g, se=se, dvec=dvec, ld=ld):
                        sg = shape_edge(g)[0] @ se                    # host-edge parameter of the Gauss point
                        return dvec if ld["s0"] <= sg <= ld["s1"] else np.zeros(2)
                    _edge_force(Fl, nodes, (a, b, m), uniform)
        elif ld["kind"] == "ring":
            ps = prob["passive"][ld["passive"]]
            c, r = np.asarray(ps["centre"], float), ps["radius"]
            dvec = np.asarray(ld["dir"], float) / np.linalg.norm(ld["dir"])
            p0 = 2 * ld["mag"] / (np.pi * r)

            def bearing(x, g, c=c, r=r, dvec=dvec, p0=p0):
                er = (x - c) / max(np.linalg.norm(x - c), 1e-300)
                return p0 * max(er @ dvec, 0.0) * er
            rr = np.hypot(nodes[:, 0] - c[0], nodes[:, 1] - c[1])
            for a, b, m in bedges:
                if all(abs(rr[i] - r) < tol for i in (a, b, m)):
                    _edge_force(Fl, nodes, (a, b, m), bearing)
        res = np.linalg.norm([Fl[0::2].sum(), Fl[1::2].sum()])
        if res <= 0:
            raise RuntimeError(f"no boundary edge found for the load {ld}")
        F += Fl * (ld["mag"] / res)                            # exact resultant (quadrature of the segment ends)
    # floating islands: mesh components without a supported node (slivers of material cut off by a loop that
    # protrudes beyond the square) carry rigid-body modes; their nodes are fixed and reported so that samples on
    # them can be excluded. An island that carries load has no body-fitted solution (broken load path).
    from scipy.sparse.csgraph import connected_components
    adj = sp.coo_matrix((np.ones(6 * len(tris)), (np.repeat(tris[:, 0], 6), tris.ravel())), shape=(n, n))
    _, label = connected_components(adj, directed=False)
    supported = np.zeros(n, bool)
    supported[np.array([d // 2 for d in fixed], int)] = True
    island_nodes = ~np.isin(label, np.unique(label[supported]))
    if island_nodes.any():
        f_isl = np.linalg.norm(F.reshape(n, 2)[island_nodes])
        if f_isl > 1e-9 * np.linalg.norm(F):
            raise RuntimeError(f"load on a floating island ({island_nodes.sum()} nodes): broken load path")
        fixed.update(2 * i + c for i in np.flatnonzero(island_nodes) for c in (0, 1))
    fixed = np.array(sorted(fixed), int)
    free = np.setdiff1d(np.arange(2 * n), fixed)
    U = np.zeros(2 * n)
    lu = splu(K[free][:, free].tocsc())
    U[free] = lu.solve(F[free])
    return {"U": U, "J": float(F @ U), "V": float(area / host.area), "fixed_dofs": len(fixed),
            "boundary_edges": len(bedges), "islands": int(len(np.unique(label[island_nodes]))),
            "island_nodes": island_nodes}


def _edge_force(F, nodes, edge, traction_fn):
    """Consistent nodal forces of a traction on a quadratic boundary edge (3-point Gauss); traction_fn receives the
    physical point and the edge coordinate g in [-1, 1] of every Gauss point."""
    Xe = nodes[list(edge)]
    for g, w in zip(G3, W3):
        N, dN = shape_edge(g)
        x = N @ Xe
        ds = np.linalg.norm(dN @ Xe)
        t = traction_fn(x, g)
        for k, i in enumerate(edge):
            F[2 * i] += N[k] * t[0] * ds * w
            F[2 * i + 1] += N[k] * t[1] * ds * w


def structured_p2(L, H, nx, ny):
    """Structured quadratic triangulation of the rectangle [0, L] x [0, H] (gmsh node order), for tests."""
    xs, ys = np.linspace(0, L, 2 * nx + 1), np.linspace(0, H, 2 * ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel()], 1)
    nid = lambda i, j: i * (2 * ny + 1) + j  # noqa: E731
    tris = []
    for i in range(nx):
        for j in range(ny):
            a, b = 2 * i, 2 * j
            c00, c20, c22, c02 = nid(a, b), nid(a + 2, b), nid(a + 2, b + 2), nid(a, b + 2)
            tris.append([c00, c20, c22, nid(a + 1, b), nid(a + 2, b + 1), nid(a + 1, b + 1)])
            tris.append([c00, c22, c02, nid(a + 1, b + 1), nid(a + 1, b + 2), nid(a, b + 1)])
    return nodes, np.array(tris)


def nodal_stress(nodes, tris, U):
    """Von Mises stress at the nodes, averaged over the adjacent elements (stress evaluated at the nodes)."""
    D = plane_stress_D()
    ref = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]], float)
    acc, cnt = np.zeros((len(nodes), 3)), np.zeros(len(nodes))
    for tri in tris:
        X = nodes[tri]
        ue = np.stack([U[2 * tri], U[2 * tri + 1]], 1).ravel()
        for k, (xi, eta) in enumerate(ref):
            _, dN = shape_p2(xi, eta)
            dNx = dN @ np.linalg.inv(dN.T @ X).T
            B = np.zeros((3, 12))
            B[0, 0::2], B[1, 1::2] = dNx[:, 0], dNx[:, 1]
            B[2, 0::2], B[2, 1::2] = dNx[:, 1], dNx[:, 0]
            acc[tri[k]] += D @ (B @ ue)
            cnt[tri[k]] += 1
    s = acc / np.maximum(cnt, 1)[:, None]
    return np.sqrt(np.maximum(s[:, 0] ** 2 - s[:, 0] * s[:, 1] + s[:, 1] ** 2 + 3 * s[:, 2] ** 2, 0.0))


def boundary_values(nodes, tris, values, points):
    """Values at points on the boundary, interpolated with the quadratic shape functions of the nearest boundary
    edge; returns the values and the distance of every point to the mesh boundary."""
    bedges = boundary_edges(tris)
    best = np.full(len(points), np.inf)
    out = np.zeros(len(points))
    ss = np.linspace(-1, 1, 21)
    for a, b, m in bedges:
        Xe = nodes[[a, b, m]]
        Ns = np.stack([shape_edge(s)[0] for s in ss])              # (21, 3)
        Xs = Ns @ Xe
        d = np.linalg.norm(points[:, None, :] - Xs[None], axis=2)
        j = np.argmin(d, 1)
        dj = d[np.arange(len(points)), j]
        better = dj < best
        best[better] = dj[better]
        out[better] = (Ns[j] @ values[[a, b, m]])[better]
    return out, best


def reference(npz_path, step_path, size=None):
    d = np.load(npz_path, allow_pickle=False)
    prob, met = json.loads(str(d["problem"])), json.loads(str(d["metrics"]))
    if len(prob["host"]["P"][0][0]) != 2:
        raise ValueError("body-fitted references are implemented for planar designs only")
    host = Host.from_dict(prob["host"]).build(8, 8)
    h = np.sqrt(host.area / (met["nel"][0] * met["nel"][1]))
    size = 0.5 * h if size is None else size
    nodes, tris = mesh_face(step_path, size)
    sol = solve_face(prob, nodes, tris)
    vm = nodal_stress(nodes, tris, sol["U"])
    out = {"nodes": nodes, "tris": tris, "U": sol["U"], "J": sol["J"], "V": sol["V"], "mesh_size": size,
           "n_nodes": len(nodes), "n_elements": len(tris), "fixed_dofs": sol["fixed_dofs"], "vm_nodes": vm,
           "J_eval": met["J_final"], "V_eval": met["V_final"]}
    if "loop_x" in d.files and d["loop_x"].size:
        X = d["loop_x"].reshape(-1, 2)
        vals, dist = boundary_values(nodes, tris, vm, X)
        out["loop_vm_bodyfit"] = vals.reshape(d["loop_x"].shape[:2])
        out["loop_dist_to_mesh"] = dist.reshape(d["loop_x"].shape[:2])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("step")
    ap.add_argument("--size", type=float, default=None, help="mesh size (default: half the optimization element)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = reference(args.npz, args.step, args.size)
    print(json.dumps({k: v for k, v in res.items() if np.isscalar(v)}, indent=1, default=float))
    if args.out:
        np.savez_compressed(args.out, **res)
