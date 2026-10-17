"""STEP export of a trimmed design: the host NURBS surface with the loops as pcurves (no approximation of loops).

Holes are cut in the parameter square by Booleans; every resulting edge is then rebuilt as a curve on the host
surface. Planar hosts are written as faces in the plane z = 0, shells as faces on their 3D surface.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot  # noqa: E402,F401  (import before OCP: OCP's libexpat otherwise breaks pyexpat)
import numpy as np  # noqa: E402

try:
    from OCP.BRep import BRep_Builder
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon,
                                    BRepBuilderAPI_MakeWire)
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepLib import BRepLib
    from OCP.BRepTools import BRepTools, BRepTools_WireExplorer
    from OCP.GC import GC_MakeEllipse
    from OCP.Geom import Geom_BSplineCurve, Geom_BSplineSurface
    from OCP.GeomAPI import GeomAPI
    from OCP.GProp import GProp_GProps
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.ShapeFix import ShapeFix_Shape
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
    try:                                                              # OCP 7.x
        from OCP.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
        from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal, TColStd_Array2OfReal
    except ImportError:                                               # OCP 8.x
        from OCP.collections import Array1_double as TColStd_Array1OfReal
        from OCP.collections import Array1_gp_Pnt as TColgp_Array1OfPnt
        from OCP.collections import Array1_int as TColStd_Array1OfInteger
        from OCP.collections import Array2_double as TColStd_Array2OfReal
        from OCP.collections import Array2_gp_Pnt as TColgp_Array2OfPnt
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS, TopoDS_Compound
    from OCP.gp import gp_Ax2, gp_Ax3, gp_Dir, gp_Elips, gp_Pln, gp_Pnt
    HAVE_OCP = True
except Exception:  # noqa: BLE001
    HAVE_OCP = False


def _knots(U):
    U = np.asarray(U, float)
    uk, mult = np.unique(np.round(U, 12), return_counts=True)
    k, m = TColStd_Array1OfReal(1, len(uk)), TColStd_Array1OfInteger(1, len(uk))
    for i, (a, b) in enumerate(zip(uk, mult)):
        k.SetValue(i + 1, float(a))
        m.SetValue(i + 1, int(b))
    return k, m


def host_surface(hd):
    P, W = np.asarray(hd["P"], float), np.asarray(hd["W"], float)
    ncu, ncv = W.shape
    poles, wts = TColgp_Array2OfPnt(1, ncu, 1, ncv), TColStd_Array2OfReal(1, ncu, 1, ncv)
    for i in range(ncu):
        for j in range(ncv):
            p = P[i, j]
            poles.SetValue(i + 1, j + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0))
            wts.SetValue(i + 1, j + 1, float(W[i, j]))
    ku, mu = _knots(hd["Uu"])
    kv, mv = _knots(hd["Uv"])
    return Geom_BSplineSurface(poles, wts, ku, kv, mu, mv, 2, 2)


def _periodic_curve(C):
    n = len(C)
    poles = TColgp_Array1OfPnt(1, n)
    for i, p in enumerate(C):
        poles.SetValue(i + 1, gp_Pnt(float(p[0]), float(p[1]), 0.0))
    knots, mults = TColStd_Array1OfReal(1, n + 1), TColStd_Array1OfInteger(1, n + 1)
    for i in range(n + 1):
        knots.SetValue(i + 1, i / n)
        mults.SetValue(i + 1, 1)
    return Geom_BSplineCurve(poles, knots, mults, 3, True)


def _face_of_edge(edge):
    return BRepBuilderAPI_MakeFace(BRepBuilderAPI_MakeWire(edge).Wire(), True).Face()


def _passive_face(ps, hd):
    if ps["kind"] == "polygon":
        poly = BRepBuilderAPI_MakePolygon()
        for p in ps["uv"]:
            poly.Add(gp_Pnt(float(p[0]), float(p[1]), 0.0))
        poly.Close()
        return BRepBuilderAPI_MakeFace(poly.Wire(), True).Face()
    L, H = hd["params"]["L"], hd["params"]["H"]                    # rectangle host: circle -> exact ellipse in (u, v)
    cu, cv = ps["centre"][0] / L, ps["centre"][1] / H
    ru, rv = ps["radius"] / L, ps["radius"] / H
    xdir = gp_Dir(1, 0, 0) if ru >= rv else gp_Dir(0, 1, 0)
    el = gp_Elips(gp_Ax2(gp_Pnt(cu, cv, 0), gp_Dir(0, 0, 1), xdir), max(ru, rv), min(ru, rv))
    return _face_of_edge(BRepBuilderAPI_MakeEdge(GC_MakeEllipse(el).Value()).Edge())


def _faces(sh):
    ex, out = TopExp_Explorer(sh, TopAbs_FACE), []
    while ex.More():
        out.append(TopoDS.Face_s(ex.Current()) if hasattr(TopoDS, "Face_s") else TopoDS.Face(ex.Current()))
        ex.Next()
    return out


def _as_wire(x):
    return TopoDS.Wire_s(x) if hasattr(TopoDS, "Wire_s") else TopoDS.Wire(x)


def _area(sh):
    pr = GProp_GProps()
    BRepGProp.SurfaceProperties_s(sh, pr, 1e-9)
    return abs(pr.Mass())


class _Quiet:
    """Redirect the C-level stdout (OpenCASCADE transfer statistics) to /dev/null."""

    def __enter__(self):
        import sys
        sys.stdout.flush()
        self.fd = os.dup(1)
        self.null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.null, 1)

    def __exit__(self, *exc):
        try:
            import ctypes
            ctypes.CDLL(None).fflush(None)                              # flush C stdio while redirected
        except Exception:  # noqa: BLE001
            pass
        os.dup2(self.fd, 1)
        os.close(self.fd)
        os.close(self.null)


def export_design(prob, loops_uv, path):
    """Write the trimmed face(s) to STEP, read them back and check validity and area."""
    with _Quiet():
        return _export(prob, loops_uv, path)


def _export(prob, loops_uv, path):
    hd = prob["host"]
    pln = gp_Pln(gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1), gp_Dir(1, 0, 0)))
    sq = BRepBuilderAPI_MakeFace(pln, 0.0, 1.0, 0.0, 1.0).Face()
    for C in loops_uv:
        sq = BRepAlgoAPI_Cut(sq, _face_of_edge(BRepBuilderAPI_MakeEdge(_periodic_curve(C)).Edge())).Shape()
    for ps in prob["passive"]:
        sq = BRepAlgoAPI_Cut(sq, _passive_face(ps, hd)).Shape()
    S = host_surface(hd)
    comp, bb = TopoDS_Compound(), BRep_Builder()
    bb.MakeCompound(comp)
    n_faces = n_edges = 0
    for f in _faces(sq):
        outer = BRepTools.OuterWire_s(f)
        wires, exw = [], TopExp_Explorer(f, TopAbs_WIRE)
        while exw.More():
            wires.append(_as_wire(exw.Current()))
            exw.Next()
        wires.sort(key=lambda w_: 0 if w_.IsSame(outer) else 1)
        built = []
        for wr in wires:
            mw, we = BRepBuilderAPI_MakeWire(), BRepTools_WireExplorer(wr, f)
            while we.More():
                e = we.Current()
                ad = BRepAdaptor_Curve(e)
                c2d = GeomAPI.To2d_s(ad.Curve().Curve(), pln)
                ne = BRepBuilderAPI_MakeEdge(c2d, S, ad.FirstParameter(), ad.LastParameter()).Edge()
                if e.Orientation() == TopAbs_REVERSED:
                    ne.Reverse()
                mw.Add(ne)
                n_edges += 1
                we.Next()
            built.append(mw.Wire())
        mf = BRepBuilderAPI_MakeFace(S, built[0], True)
        for wr in built[1:]:
            mf.Add(wr)
        face = mf.Face()
        BRepLib.BuildCurves3d_s(face)
        fix = ShapeFix_Shape(face)
        fix.Perform()
        bb.Add(comp, fix.Shape())
        n_faces += 1
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = STEPControl_Writer()
    w.Transfer(comp, STEPControl_AsIs)
    w.Write(path)
    r = STEPControl_Reader()
    ok = r.ReadFile(path) == IFSelect_RetDone
    r.TransferRoots()
    back = r.OneShape()
    return {"faces": n_faces, "pcurve_edges": n_edges, "valid": bool(ok and BRepCheck_Analyzer(back).IsValid()),
            "area": _area(back), "bytes": os.path.getsize(path)}
