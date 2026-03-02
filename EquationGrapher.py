"""
Equation Grapher — Complete Function Study with Sign Tables
Requires: pip install matplotlib sympy numpy
Run: python equation_grapher.py
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import re, warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(s):
    s = s.strip()
    s = re.sub(r'^[yY]\s*=\s*', '', s)
    s = re.sub(r'^f\s*\(\s*[xX]\s*\)\s*=\s*', '', s)
    s = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', s)
    s = re.sub(r'([a-zA-Z\d])\s*\^', r'\1**', s)
    s = re.sub(r'\)\s*\(', r')*(', s)
    return s

def get_x(expr):
    """Get the actual x symbol from the expression (not our declared one)."""
    for s in expr.free_symbols:
        if s.name == 'x':
            return s
    return None

def safe_float(v):
    try:
        c = complex(v)
        return float(c.real) if abs(c.imag) < 1e-9 else None
    except:
        return None

# ══════════════════════════════════════════════════════════════════════════════
#  SIGN TABLE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_sign_table(func_expr, x_sym, key_points, label):
    """
    Build a Unicode sign table for func_expr over intervals defined by key_points.
    Returns a list of lines.
    """
    # key_points: sorted list of real floats (roots, critical points, etc.)
    if not key_points:
        # just check one point
        try:
            val = float(func_expr.subs(x_sym, 0))
            sign = "+" if val > 0 else ("-" if val < 0 else "0")
        except:
            sign = "?"
        return [f"  {label}: {sign} on all of ℝ"]

    pts_sorted = sorted(set(key_points))
    # Build column headers: -∞ | p1 | p2 | ... | +∞
    cols = []
    cols.append("-∞")
    for p in pts_sorted:
        cols.append(f"{sp.nsimplify(p, rational=True)}")
    cols.append("+∞")

    # Evaluate sign in each open interval
    signs = []
    test_points = []
    # before first
    test_points.append(pts_sorted[0] - 1)
    # between
    for i in range(len(pts_sorted) - 1):
        test_points.append((pts_sorted[i] + pts_sorted[i+1]) / 2)
    # after last
    test_points.append(pts_sorted[-1] + 1)

    interval_signs = []
    for tp in test_points:
        try:
            val = float(func_expr.subs(x_sym, tp))
            if val > 1e-12:
                interval_signs.append("+")
            elif val < -1e-12:
                interval_signs.append("-")
            else:
                interval_signs.append("0")
        except:
            interval_signs.append("?")

    # Value at each key point
    point_vals = []
    for p in pts_sorted:
        try:
            val = float(func_expr.subs(x_sym, p))
            if abs(val) < 1e-12:
                point_vals.append("0")
            elif val > 0:
                point_vals.append("+")
            else:
                point_vals.append("-")
        except:
            point_vals.append("║")  # discontinuity

    # Interleave: sign | point | sign | point | ... | sign
    row_data = []  # list of (text, is_keypoint)
    row_data.append((interval_signs[0], False))
    for i, p in enumerate(pts_sorted):
        row_data.append((point_vals[i], True))
        row_data.append((interval_signs[i+1], False))

    # Column widths
    col_widths = []
    col_widths.append(max(len("-∞"), len(interval_signs[0])) + 2)
    for i, p in enumerate(pts_sorted):
        w = max(len(str(sp.nsimplify(p, rational=True))), len(point_vals[i])) + 2
        col_widths.append(w)
        col_widths.append(max(3, len(interval_signs[i+1])) + 2)
    col_widths.append(max(len("+∞"), len(interval_signs[-1])) + 2)

    # Build header row (x values)
    header = "  x   │"
    header += f"{'−∞':^5}│"
    for p in pts_sorted:
        ps = str(sp.nsimplify(p, rational=True))
        header += f"{ps:^7}│"
        header += f"{'':^5}│"
    header += f"{'＋∞':^5}│"

    # Build sign row
    sign_row = f"  {label:4s}│"
    sign_row += f"{interval_signs[0]:^5}│"
    for i, p in enumerate(pts_sorted):
        sign_row += f"{point_vals[i]:^7}│"
        if i + 1 < len(interval_signs):
            sign_row += f"{interval_signs[i+1]:^5}│"
    sign_row += f"{'':^5}│"

    # separator
    sep = "  " + "─" * (len(header) - 2)

    lines = [sep, header, sep, sign_row, sep]

    # Explanation
    expl = []
    for i, s in enumerate(interval_signs):
        if i == 0:
            interval = f"(-∞ ; {pts_sorted[0]})"
        elif i == len(interval_signs) - 1:
            interval = f"({pts_sorted[-1]} ; +∞)"
        else:
            interval = f"({pts_sorted[i-1]} ; {pts_sorted[i]})"
        if s == "+":
            expl.append(f"    {interval}: {label} > 0")
        elif s == "-":
            expl.append(f"    {interval}: {label} < 0")
    lines += expl
    return lines

# ══════════════════════════════════════════════════════════════════════════════
#  CORE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze(raw):
    proc = preprocess(raw)
    try:
        expr = sp.sympify(proc)
    except Exception as e:
        raise ValueError(f"Cannot parse: {e}")

    x = get_x(expr)
    if x is None:
        # constant
        x = sp.Symbol('x')

    R = {}
    R["expr"] = expr
    R["x"] = x

    # ── Domain ────────────────────────────────────────────────────────────────
    excluded = []
    try:
        denom = sp.denom(sp.cancel(expr))
        if denom != 1:
            for z in sp.solve(denom, x):
                f = safe_float(z)
                if f is not None:
                    excluded.append(f)
    except: pass

    try:
        dom = sp.calculus.util.continuous_domain(expr, x, sp.S.Reals)
        if dom != sp.S.Reals and not excluded:
            R["domain_str"] = str(dom)
        else:
            raise Exception()
    except:
        if excluded:
            excl = sorted(set(excluded))
            pts = ", ".join(str(sp.nsimplify(e, rational=True)) for e in excl)
            parts = []
            prev = None
            all_pts = [None] + excl + [None]
            for i in range(len(all_pts)-1):
                l = all_pts[i]; r = all_pts[i+1]
                ls = "-∞" if l is None else f"{sp.nsimplify(l,rational=True)}⁺"
                rs = "+∞" if r is None else f"{sp.nsimplify(r,rational=True)}⁻"
                parts.append(f"({ls} ; {rs})")
            R["domain_str"] = f"ℝ \\ {{{pts}}}\n    = " + " ∪ ".join(parts)
        else:
            R["domain_str"] = "ℝ  =  (-∞ ; +∞)"

    R["excluded"] = sorted(set(excluded))

    # ── Intercepts ────────────────────────────────────────────────────────────
    try:
        R["y_int"] = sp.simplify(expr.subs(x, 0))
    except:
        R["y_int"] = "undefined"

    try:
        roots_raw = sp.solve(expr, x)
        R["roots"] = [r for r in roots_raw if safe_float(r) is not None]
    except:
        R["roots"] = []

    # ── Limits ────────────────────────────────────────────────────────────────
    try: R["lim_pos"] = str(sp.limit(expr, x, sp.oo))
    except: R["lim_pos"] = "N/A"
    try: R["lim_neg"] = str(sp.limit(expr, x, -sp.oo))
    except: R["lim_neg"] = "N/A"

    # Limits at excluded points (for asymptotes)
    R["va_limits"] = {}
    for e in R["excluded"]:
        try:
            lp = sp.limit(expr, x, e, '+')
            ln = sp.limit(expr, x, e, '-')
            R["va_limits"][e] = (str(lp), str(ln))
        except:
            R["va_limits"][e] = ("?", "?")

    # ── Asymptotes ────────────────────────────────────────────────────────────
    R["horiz_asymp"] = []
    try:
        lp = sp.limit(expr, x, sp.oo)
        ln = sp.limit(expr, x, -sp.oo)
        if lp.is_finite: R["horiz_asymp"].append(f"y = {lp}  (x → +∞)")
        if ln.is_finite and ln != lp: R["horiz_asymp"].append(f"y = {ln}  (x → -∞)")
        if not R["horiz_asymp"] and not lp.is_finite:
            m = sp.limit(expr/x, x, sp.oo)
            if m.is_finite and m != 0:
                b = sp.limit(expr - m*x, x, sp.oo)
                if b.is_finite: R["horiz_asymp"].append(f"y = {m}x + {b}  (oblique)")
    except: pass

    # ── Derivatives ───────────────────────────────────────────────────────────
    try:
        f1 = sp.diff(expr, x)
        R["f1"] = f1
        R["f1_str"] = str(f1)
        R["f1_pretty"] = str(sp.simplify(f1))
    except:
        R["f1"] = None; R["f1_str"] = "N/A"; R["f1_pretty"] = "N/A"

    try:
        f2 = sp.diff(expr, x, 2)
        R["f2"] = f2
        R["f2_str"] = str(f2)
        R["f2_pretty"] = str(sp.simplify(f2))
    except:
        R["f2"] = None; R["f2_str"] = "N/A"; R["f2_pretty"] = "N/A"

    # ── Critical points ───────────────────────────────────────────────────────
    R["critical"] = []
    if R["f1"] is not None:
        try:
            cps = sp.solve(R["f1"], x)
            R["critical"] = [c for c in cps if safe_float(c) is not None]
        except: pass

    R["classified"] = []
    if R["f2"] is not None:
        for cp in R["critical"]:
            try:
                v2 = float(R["f2"].subs(x, cp))
                fy = sp.simplify(expr.subs(x, cp))
                if v2 > 0: kind = "Local MINIMUM"
                elif v2 < 0: kind = "Local MAXIMUM"
                else: kind = "Inflection/Saddle (f''=0)"
                R["classified"].append((cp, fy, kind))
            except: pass

    # ── Inflection points ─────────────────────────────────────────────────────
    R["inflection"] = []
    if R["f2"] is not None:
        try:
            ips = sp.solve(R["f2"], x)
            R["inflection"] = [i for i in ips if safe_float(i) is not None]
        except: pass

    # ── Sign tables ───────────────────────────────────────────────────────────
    # Key points for f: roots + excluded
    f_pts = [safe_float(r) for r in R["roots"]] + R["excluded"]
    f_pts = sorted(set(p for p in f_pts if p is not None))
    R["sign_f"]  = build_sign_table(expr,   x, f_pts,         "f(x)")

    f1_pts = [safe_float(c) for c in R["critical"]] + R["excluded"]
    f1_pts = sorted(set(p for p in f1_pts if p is not None))
    R["sign_f1"] = build_sign_table(R["f1"], x, f1_pts, "f'(x)") if R["f1"] is not None else ["  N/A"]

    f2_pts = [safe_float(i) for i in R["inflection"]] + R["excluded"]
    f2_pts = sorted(set(p for p in f2_pts if p is not None))
    R["sign_f2"] = build_sign_table(R["f2"], x, f2_pts, "f''(x)") if R["f2"] is not None else ["  N/A"]

    # ── Monotonicity ──────────────────────────────────────────────────────────
    R["mono"] = []
    if R["f1"] is not None:
        all_cps = sorted(set([safe_float(c) for c in R["critical"] if safe_float(c) is not None] + R["excluded"]))
        bds = [None] + all_cps + [None]
        test_xs = (
            [all_cps[0]-1] if all_cps else [0]
        ) if all_cps else []
        if all_cps:
            tps = [all_cps[0]-1] + [(all_cps[i]+all_cps[i+1])/2 for i in range(len(all_cps)-1)] + [all_cps[-1]+1]
            for i, tp in enumerate(tps):
                try:
                    v = float(R["f1"].subs(x, tp))
                    l = bds[i]; r = bds[i+1]
                    ls = "-∞" if l is None else str(sp.nsimplify(l, rational=True))
                    rs = "+∞" if r is None else str(sp.nsimplify(r, rational=True))
                    direction = "Increasing ↑" if v > 0 else ("Decreasing ↓" if v < 0 else "Constant →")
                    R["mono"].append((f"({ls} ; {rs})", direction, v))
                except: pass

    # ── Concavity ─────────────────────────────────────────────────────────────
    R["concavity"] = []
    if R["f2"] is not None:
        all_ips = sorted(set([safe_float(i) for i in R["inflection"] if safe_float(i) is not None] + R["excluded"]))
        if all_ips:
            tps = [all_ips[0]-1] + [(all_ips[i]+all_ips[i+1])/2 for i in range(len(all_ips)-1)] + [all_ips[-1]+1]
            bds = [None] + all_ips + [None]
            for i, tp in enumerate(tps):
                try:
                    v = float(R["f2"].subs(x, tp))
                    l = bds[i]; r = bds[i+1]
                    ls = "-∞" if l is None else str(sp.nsimplify(l, rational=True))
                    rs = "+∞" if r is None else str(sp.nsimplify(r, rational=True))
                    direction = "Concave UP ∪" if v > 0 else ("Concave DOWN ∩" if v < 0 else "Linear")
                    R["concavity"].append((f"({ls} ; {rs})", direction))
                except: pass

    return R

# ══════════════════════════════════════════════════════════════════════════════
#  REPORT FORMATTER
# ══════════════════════════════════════════════════════════════════════════════

def format_report(R):
    lines = []
    x = R["x"]

    def H(n, title):
        lines.append("")
        lines.append(f"{'═'*62}")
        lines.append(f"  {n}. {title}")
        lines.append(f"{'═'*62}")

    def sub(title):
        lines.append(f"\n  ── {title}")

    def li(text):
        lines.append(f"  • {text}")

    def note(text):
        lines.append(f"    ℹ {text}")

    lines.append("╔══════════════════════════════════════════════════════════╗")
    lines.append(f"║   COMPLETE STUDY OF  f(x) = {str(R['expr'])[:28]:<28}  ║")
    lines.append("╚══════════════════════════════════════════════════════════╝")

    # ── 1. Domain ─────────────────────────────────────────────────────────────
    H(1, "DOMAIN")
    li(R["domain_str"])
    if R["excluded"]:
        note(f"Excluded points (singularities): {', '.join(str(sp.nsimplify(e,rational=True)) for e in R['excluded'])}")
        for e, (lp, ln) in R["va_limits"].items():
            ep = sp.nsimplify(e, rational=True)
            lines.append(f"    lim f(x) as x→{ep}⁺ = {lp}")
            lines.append(f"    lim f(x) as x→{ep}⁻ = {ln}")
            if 'oo' in lp or 'oo' in ln:
                lines.append(f"    → Vertical asymptote: x = {ep}")

    # ── 2. Intercepts ─────────────────────────────────────────────────────────
    H(2, "INTERCEPTS")
    li(f"Y-intercept: f(0) = {R['y_int']}")
    if R["roots"]:
        for r in R["roots"]:
            rv = sp.nsimplify(r, rational=True)
            lines.append(f"  • X-intercept (zero): x = {rv}  ≈  {float(r):.4f}")
    else:
        li("No real zeros found (or cannot solve analytically)")

    # ── 3. Limits & Asymptotes ────────────────────────────────────────────────
    H(3, "LIMITS AND ASYMPTOTES")
    li(f"lim f(x) as x → +∞  =  {R['lim_pos']}")
    li(f"lim f(x) as x → -∞  =  {R['lim_neg']}")
    if R["horiz_asymp"]:
        for a in R["horiz_asymp"]:
            li(f"Asymptote: {a}")
    else:
        li("No horizontal or oblique asymptotes")
    if R["excluded"]:
        for e in R["excluded"]:
            li(f"Vertical asymptote: x = {sp.nsimplify(e,rational=True)}")

    # ── 4. Sign of f(x) ───────────────────────────────────────────────────────
    H(4, "SIGN OF  f(x)")
    note("Shows where f(x) is positive (+), negative (-), or zero (0)")
    for l in R["sign_f"]:
        lines.append(l)

    # ── 5. First derivative ───────────────────────────────────────────────────
    H(5, "FIRST DERIVATIVE  f'(x)")
    li(f"f'(x) = {R['f1_pretty']}")
    note("f'(x) > 0 → f increasing  |  f'(x) < 0 → f decreasing  |  f'(x)=0 → critical point")
    if R["critical"]:
        for cp in R["critical"]:
            cpv = sp.nsimplify(cp, rational=True)
            lines.append(f"  • Critical point: x = {cpv}  ≈  {float(cp):.4f}")
    else:
        li("No critical points")

    sub("Sign of f'(x)")
    for l in R["sign_f1"]:
        lines.append(l)

    sub("Monotonicity (variation)")
    if R["mono"]:
        for interval, direction, slope in R["mono"]:
            lines.append(f"  • {interval:25s} → {direction}")
    else:
        li("Could not determine (no critical points)")

    # ── 6. Critical point classification ──────────────────────────────────────
    H(6, "CLASSIFICATION OF CRITICAL POINTS  (2nd derivative test)")
    note("f''(cp) > 0 → Local Minimum  |  f''(cp) < 0 → Local Maximum  |  f''(cp)=0 → inconclusive")
    if R["classified"]:
        for cp, fv, kind in R["classified"]:
            cpv = sp.nsimplify(cp, rational=True)
            lines.append(f"  • x = {cpv}  →  f(x) = {fv}")
            lines.append(f"    Classification: {kind}")
    else:
        li("No critical points to classify")

    # ── 7. Second derivative ──────────────────────────────────────────────────
    H(7, "SECOND DERIVATIVE  f''(x)")
    li(f"f''(x) = {R['f2_pretty']}")
    note("f''(x) > 0 → concave up ∪  |  f''(x) < 0 → concave down ∩")
    if R["inflection"]:
        for ip in R["inflection"]:
            ipv = sp.nsimplify(ip, rational=True)
            lines.append(f"  • Inflection point: x = {ipv}  ≈  {float(ip):.4f}")
    else:
        li("No inflection points found")

    sub("Sign of f''(x)")
    for l in R["sign_f2"]:
        lines.append(l)

    sub("Concavity")
    if R["concavity"]:
        for interval, direction in R["concavity"]:
            lines.append(f"  • {interval:25s} → {direction}")
    else:
        li("Could not determine (no inflection points)")

    # ── 8. Summary table ──────────────────────────────────────────────────────
    H(8, "SUMMARY — VARIATION TABLE")
    note("↑ = increasing, ↓ = decreasing, ∪ = concave up, ∩ = concave down")

    all_xpts = sorted(set(
        [safe_float(c) for c in R["critical"] if safe_float(c) is not None] +
        [safe_float(i) for i in R["inflection"] if safe_float(i) is not None] +
        R["excluded"]
    ))

    if all_xpts:
        header = "  x     │ -∞ "
        for p in all_xpts:
            pv = sp.nsimplify(p, rational=True)
            header += f"│{str(pv):^7}"
        header += "│ +∞ │"
        lines.append("  " + "─"*len(header))
        lines.append(header)
        lines.append("  " + "─"*len(header))

        def get_sign_in_interval(func, left, right):
            if func is None: return "?"
            tp = (left + right) / 2 if (left is not None and right is not None) else \
                 (right - 1 if left is None else left + 1)
            try:
                v = float(func.subs(x, tp))
                return "+" if v > 0 else ("-" if v < 0 else "0")
            except:
                return "?"

        bds = [None] + all_xpts + [None]

        # f' row
        row = "  f'(x) │ "
        for i in range(len(bds)-1):
            l = bds[i]; r = bds[i+1]
            tp = (l if l is not None else r-1) + 0.0001 if l is not None else (r-1 if r is not None else 0)
            tp = l + 0.5 if (l is not None and r is not None) else (r - 1 if l is None else l + 1)
            s = get_sign_in_interval(R["f1"], bds[i], bds[i+1]) if R["f1"] is not None else "?"
            if r is not None and r in R["excluded"]:
                row += f"  {s}   │  ║  "
            elif r is not None:
                val_str = "0"
                row += f"  {s}   │{val_str:^7}"
            else:
                row += f"  {s}   │"
        row += "    │"
        lines.append(row)

        # f'' row
        row2 = "  f''(x)│ "
        for i in range(len(bds)-1):
            s = get_sign_in_interval(R["f2"], bds[i], bds[i+1]) if R["f2"] is not None else "?"
            r = bds[i+1]
            if r is not None and r in R["excluded"]:
                row2 += f"  {s}   │  ║  "
            elif r is not None:
                row2 += f"  {s}   │       "
            else:
                row2 += f"  {s}   │"
        row2 += "    │"
        lines.append(row2)
        lines.append("  " + "─"*len(header))
    else:
        li("No notable points found — function has no critical or inflection points on ℝ")

    lines.append("")
    lines.append("═"*62)
    lines.append("  END OF STUDY")
    lines.append("═"*62)
    return "\n".join(lines)

# ══════════════════════════════════════════════════════════════════════════════
#  COLORS
# ══════════════════════════════════════════════════════════════════════════════
BG      = "#0e0e1a"
PANEL   = "#14142a"
ACCENT  = "#9b6dff"
ACCENT2 = "#e040fb"
FG      = "#ddd8f0"
ENTRY   = "#1a1a30"
GREEN   = "#50fa7b"
ORANGE  = "#ffb86c"
RED     = "#ff5555"
CYAN    = "#8be9fd"
YELLOW  = "#f1fa8c"

# ══════════════════════════════════════════════════════════════════════════════
#  GUI
# ══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("∫ Equation Grapher — Complete Function Study")
        self.configure(bg=BG)
        self.geometry("1400x860")
        self.minsize(1000, 650)
        self._R = None
        self._xs = self._ys = None
        self._auto_y = tk.BooleanVar(value=True)
        self._build()

    def _build(self):
        # ── Top bar ───────────────────────────────────────────────────────────
        top = tk.Frame(self, bg=BG, pady=6)
        top.pack(fill=tk.X, padx=16)

        tk.Label(top, text="∫", font=("Georgia", 22, "bold"), bg=BG, fg=ACCENT2).pack(side=tk.LEFT)
        tk.Label(top, text="  f(x) =", font=("Courier New", 14, "bold"), bg=BG, fg=ACCENT).pack(side=tk.LEFT)

        self.entry = tk.Entry(top, font=("Courier New", 14), bg=ENTRY, fg=FG,
                              insertbackground=FG, relief=tk.FLAT, bd=8, width=32)
        self.entry.insert(0, "3*x**4+3*x-3")
        self.entry.pack(side=tk.LEFT, padx=8)
        self.entry.bind("<Return>", lambda e: self._run())

        for label, text in [("x min:", "xmin"), ("x max:", "xmax")]:
            tk.Label(top, text=label, font=("Courier New", 10), bg=BG, fg=FG).pack(side=tk.LEFT, padx=(8,2))
            e = tk.Entry(top, font=("Courier New", 11), bg=ENTRY, fg=FG,
                         insertbackground=FG, relief=tk.FLAT, bd=4, width=6)
            e.insert(0, "-10" if "min" in label else "10")
            e.pack(side=tk.LEFT, padx=2)
            setattr(self, text, e)

        tk.Label(top, text="y min:", font=("Courier New", 10), bg=BG, fg="#aaa8cc").pack(side=tk.LEFT, padx=(10,2))
        self.ymin = tk.Entry(top, font=("Courier New", 11), bg=ENTRY, fg=FG,
                             insertbackground=FG, relief=tk.FLAT, bd=4, width=6)
        self.ymin.pack(side=tk.LEFT, padx=2)
        self.ymin.bind("<Return>", lambda e: self._apply_ylim())

        tk.Label(top, text="y max:", font=("Courier New", 10), bg=BG, fg="#aaa8cc").pack(side=tk.LEFT, padx=(4,2))
        self.ymax = tk.Entry(top, font=("Courier New", 11), bg=ENTRY, fg=FG,
                             insertbackground=FG, relief=tk.FLAT, bd=4, width=6)
        self.ymax.pack(side=tk.LEFT, padx=2)
        self.ymax.bind("<Return>", lambda e: self._apply_ylim())

        self._auto_y = tk.BooleanVar(value=True)
        self._auto_y_btn = tk.Button(top, text="Auto Y ✓", font=("Courier New", 9, "bold"),
                                     bg="#1e1e40", fg=CYAN, relief=tk.FLAT, padx=6, pady=3,
                                     cursor="hand2", command=self._toggle_auto_y)
        self._auto_y_btn.pack(side=tk.LEFT, padx=4)

        tk.Button(top, text="▶  ANALYZE", font=("Courier New", 12, "bold"),
                  bg=ACCENT, fg="white", activebackground=ACCENT2,
                  relief=tk.FLAT, padx=14, pady=3, cursor="hand2",
                  command=self._run).pack(side=tk.LEFT, padx=12)

        tk.Button(top, text="✕", font=("Courier New", 11),
                  bg=PANEL, fg=FG, relief=tk.FLAT, padx=8, pady=3,
                  cursor="hand2", command=self._clear).pack(side=tk.LEFT)

        # ── Examples ──────────────────────────────────────────────────────────
        ex = tk.Frame(self, bg=BG)
        ex.pack(fill=tk.X, padx=16, pady=(0, 6))
        tk.Label(ex, text="Examples:", font=("Courier New", 9), bg=BG, fg="#666").pack(side=tk.LEFT, padx=(0,6))
        for label, expr in [
            ("3x⁴+3x−3", "3*x**4+3*x-3"),
            ("1/x",       "1/x"),
            ("1/(x²−1)", "1/(x**2-1)"),
            ("x³−3x",    "x**3-3*x"),
            ("sin(x)",   "sin(x)"),
            ("ln(x)",    "log(x)"),
            ("e^(−x²)",  "exp(-x**2)"),
            ("√x",       "sqrt(x)"),
        ]:
            tk.Button(ex, text=label, font=("Courier New", 9),
                      bg=PANEL, fg=CYAN, relief=tk.FLAT, padx=5, pady=2,
                      cursor="hand2",
                      command=lambda e=expr: self._load(e)).pack(side=tk.LEFT, padx=2)

        # ── Main paned area ───────────────────────────────────────────────────
        pw = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=BG, sashwidth=6)
        pw.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,8))

        # Graph
        gf = tk.Frame(pw, bg=PANEL)
        pw.add(gf, width=700)
        self.fig = Figure(facecolor=PANEL, edgecolor=PANEL)
        self.ax = self.fig.add_subplot(111)
        self._style_ax()
        self.canvas = FigureCanvasTkAgg(self.fig, master=gf)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        tb = NavigationToolbar2Tk(self.canvas, gf)
        tb.config(bg=PANEL)
        tb.update()
        self._annot = self.ax.annotate("", xy=(0,0), xytext=(14,14),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="#0e0e1a", ec=ACCENT2, lw=1.2),
            arrowprops=dict(arrowstyle="->", color=ACCENT2),
            color=FG, fontsize=9, visible=False)
        self.canvas.mpl_connect("motion_notify_event", self._hover)

        # Study panel with tabs
        sf = tk.Frame(pw, bg=PANEL)
        pw.add(sf, width=620)

        tab_bar = tk.Frame(sf, bg=BG)
        tab_bar.pack(fill=tk.X)
        self._tabs = {}
        self._tab_frames = {}
        self._active_tab = tk.StringVar(value="study")

        for tid, tlabel in [("study", "📋 Full Study"), ("sign", "± Sign Tables"), ("deriv", "∂ Derivatives")]:
            btn = tk.Button(tab_bar, text=tlabel, font=("Courier New", 10, "bold"),
                            bg=ACCENT if tid=="study" else PANEL,
                            fg="white" if tid=="study" else FG,
                            relief=tk.FLAT, padx=10, pady=4,
                            cursor="hand2",
                            command=lambda t=tid: self._switch_tab(t))
            btn.pack(side=tk.LEFT, padx=2, pady=2)
            self._tabs[tid] = btn

        content = tk.Frame(sf, bg=PANEL)
        content.pack(fill=tk.BOTH, expand=True)

        for tid in ["study", "sign", "deriv"]:
            f = tk.Frame(content, bg=PANEL)
            txt = scrolledtext.ScrolledText(f, font=("Courier New", 9),
                                            bg=PANEL, fg=FG,
                                            insertbackground=FG, relief=tk.FLAT,
                                            padx=10, pady=8, state=tk.DISABLED,
                                            wrap=tk.NONE,
                                            selectbackground=ACCENT)
            txt.pack(fill=tk.BOTH, expand=True)
            # color tags
            txt.tag_config("header",  foreground=ACCENT2, font=("Courier New", 10, "bold"))
            txt.tag_config("subhead", foreground=CYAN, font=("Courier New", 9, "bold"))
            txt.tag_config("pos",     foreground=GREEN)
            txt.tag_config("neg",     foreground=RED)
            txt.tag_config("zero",    foreground=YELLOW)
            txt.tag_config("note",    foreground=ORANGE)
            txt.tag_config("label",   foreground=CYAN)
            self._tab_frames[tid] = (f, txt)

        self._switch_tab("study")

        # Status
        self.status = tk.Label(self, font=("Courier New", 9), bg=BG, fg="#666",
                               anchor=tk.W, padx=14,
                               text="Type a function and press ANALYZE  (or hit Enter)")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _switch_tab(self, tid):
        for t, (f, _) in self._tab_frames.items():
            f.pack_forget()
        self._tab_frames[tid][0].pack(fill=tk.BOTH, expand=True)
        for t, btn in self._tabs.items():
            btn.config(bg=ACCENT if t==tid else PANEL,
                       fg="white" if t==tid else FG)
        self._active_tab.set(tid)

    def _style_ax(self):
        self.ax.set_facecolor("#0a0a18")
        self.ax.tick_params(colors=FG, labelsize=8)
        for s in self.ax.spines.values():
            s.set_edgecolor("#2a2a44")
        self.ax.grid(True, color="#1a1a2e", linestyle="--", linewidth=0.5, alpha=0.9)
        self.ax.axhline(0, color="#333355", linewidth=0.9)
        self.ax.axvline(0, color="#333355", linewidth=0.9)
        self.fig.tight_layout(pad=1.5)

    def _load(self, expr):
        self.entry.delete(0, tk.END)
        self.entry.insert(0, expr)
        self._run()

    def _clear(self):
        self.entry.delete(0, tk.END)
        self.ax.cla(); self._style_ax(); self.canvas.draw()
        for _, txt in self._tab_frames.values():
            txt.config(state=tk.NORMAL); txt.delete("1.0", tk.END); txt.config(state=tk.DISABLED)
        self.status.config(text="Cleared.")

    def _run(self):
        raw = self.entry.get().strip()
        if not raw: return
        try:
            xmin = float(self.xmin.get())
            xmax = float(self.xmax.get())
        except:
            messagebox.showerror("Error", "x min/max must be numbers"); return

        self.status.config(text="⏳  Computing…"); self.update_idletasks()

        try:
            R = analyze(raw)
        except Exception as e:
            messagebox.showerror("Parse Error", str(e))
            self.status.config(text="Error."); return

        self._R = R
        report = format_report(R)
        self._fill_study(report)
        self._fill_sign(R)
        self._fill_deriv(R)
        self._plot(R, xmin, xmax)
        # Sync y fields from actual axis limits after auto-plot
        if self._auto_y.get():
            ylo, yhi = self.ax.get_ylim()
            self.ymin.delete(0, tk.END); self.ymin.insert(0, f"{ylo:.2f}")
            self.ymax.delete(0, tk.END); self.ymax.insert(0, f"{yhi:.2f}")
        self.status.config(text=f"✓  f(x) = {R['expr']}   |  hover graph for values  |  use tabs for details")

    # ── Fill study tab ────────────────────────────────────────────────────────
    def _fill_study(self, text):
        _, txt = self._tab_frames["study"]
        txt.config(state=tk.NORMAL); txt.delete("1.0", tk.END)
        for line in text.split("\n"):
            if "═" in line or "╔" in line or "╚" in line or "╗" in line or "╝" in line:
                txt.insert(tk.END, line+"\n", "header")
            elif line.strip().startswith("──"):
                txt.insert(tk.END, line+"\n", "subhead")
            elif "ℹ" in line:
                txt.insert(tk.END, line+"\n", "note")
            elif "+" in line and ("→" in line or "│" in line):
                txt.insert(tk.END, line+"\n", "pos")
            elif line.strip().startswith("•"):
                txt.insert(tk.END, line+"\n", "label")
            else:
                txt.insert(tk.END, line+"\n")
        txt.config(state=tk.DISABLED)

    # ── Fill sign tab ─────────────────────────────────────────────────────────
    def _fill_sign(self, R):
        _, txt = self._tab_frames["sign"]
        txt.config(state=tk.NORMAL); txt.delete("1.0", tk.END)

        def section(title, lines_data):
            txt.insert(tk.END, f"\n{'═'*60}\n  {title}\n{'═'*60}\n", "header")
            for l in lines_data:
                if "+" in l:
                    txt.insert(tk.END, l+"\n", "pos")
                elif "─" in l or "│" in l:
                    txt.insert(tk.END, l+"\n", "subhead")
                elif "-" in l and "─" not in l:
                    txt.insert(tk.END, l+"\n", "neg")
                else:
                    txt.insert(tk.END, l+"\n")

        section("SIGN OF  f(x)", R["sign_f"])
        section("SIGN OF  f'(x)  →  Monotonicity", R["sign_f1"])
        section("SIGN OF  f''(x)  →  Concavity", R["sign_f2"])
        txt.config(state=tk.DISABLED)

    # ── Fill derivatives tab ──────────────────────────────────────────────────
    def _fill_deriv(self, R):
        _, txt = self._tab_frames["deriv"]
        txt.config(state=tk.NORMAL); txt.delete("1.0", tk.END)

        def H(t): txt.insert(tk.END, f"\n{'═'*60}\n  {t}\n{'═'*60}\n", "header")
        def li(t): txt.insert(tk.END, f"  • {t}\n", "label")
        def note(t): txt.insert(tk.END, f"    ℹ {t}\n", "note")

        H("FIRST DERIVATIVE  f'(x)")
        txt.insert(tk.END, f"  f'(x) = {R['f1_pretty']}\n\n")
        note("The derivative gives the instantaneous rate of change (slope of tangent)")
        note("f'(x) > 0  →  function is INCREASING on that interval")
        note("f'(x) < 0  →  function is DECREASING on that interval")
        note("f'(x) = 0  →  CRITICAL POINT (potential extremum)")
        if R["critical"]:
            txt.insert(tk.END, "\n  Critical points (f'=0):\n")
            for cp in R["critical"]:
                cpv = sp.nsimplify(cp, rational=True)
                li(f"x = {cpv}  ≈  {float(cp):.6f}")
        if R["classified"]:
            txt.insert(tk.END, "\n  Classification:\n")
            for cp, fv, kind in R["classified"]:
                cpv = sp.nsimplify(cp, rational=True)
                txt.insert(tk.END, f"  • x = {cpv}  →  f = {fv}  →  {kind}\n",
                           "pos" if "MIN" in kind else "neg" if "MAX" in kind else "zero")

        H("SECOND DERIVATIVE  f''(x)")
        txt.insert(tk.END, f"  f''(x) = {R['f2_pretty']}\n\n")
        note("The second derivative measures the curvature of the function")
        note("f''(x) > 0  →  CONCAVE UP (∪)  — curve bends upward")
        note("f''(x) < 0  →  CONCAVE DOWN (∩)  — curve bends downward")
        note("f''(x) = 0  →  potential INFLECTION POINT")
        if R["inflection"]:
            txt.insert(tk.END, "\n  Inflection points (f''=0):\n")
            for ip in R["inflection"]:
                ipv = sp.nsimplify(ip, rational=True)
                li(f"x = {ipv}  ≈  {float(ip):.6f}")
        if R["concavity"]:
            txt.insert(tk.END, "\n  Concavity by interval:\n")
            for interval, direction in R["concavity"]:
                txt.insert(tk.END, f"  • {interval:25s} → {direction}\n",
                           "pos" if "UP" in direction else "neg")

        txt.config(state=tk.DISABLED)

    # ── Plot ──────────────────────────────────────────────────────────────────
    def _plot(self, R, xmin, xmax):
        self.ax.cla(); self._style_ax()
        x = R["x"]
        expr = R["expr"]

        try:
            f_np = sp.lambdify(x, expr, modules=["numpy"])
        except:
            return

        xs = np.linspace(xmin, xmax, 3000)

        # Mask near excluded points
        mask = np.ones(len(xs), bool)
        for e in R["excluded"]:
            mask &= np.abs(xs - e) > (xmax-xmin)/500

        with np.errstate(all='ignore'):
            ys = f_np(xs)
            try: ys = np.array(ys, dtype=float)
            except: return

        ys[~mask] = np.nan
        ys[~np.isfinite(ys)] = np.nan

        # Clip for visibility
        fin = ys[np.isfinite(ys)]
        if len(fin):
            q1, q3 = np.nanpercentile(fin, 2), np.nanpercentile(fin, 98)
            spread = max(q3-q1, 1)
            ys = np.clip(ys, q1 - 4*spread, q3 + 4*spread)

        self._xs = xs; self._ys = ys

        # Main curve
        self.ax.plot(xs, ys, color=ACCENT2, linewidth=2.2, label="f(x)", zorder=4)

        # f'
        if R["f1"] is not None:
            try:
                f1_np = sp.lambdify(x, R["f1"], modules=["numpy"])
                ys1 = np.array(f1_np(xs), dtype=float)
                ys1[~mask] = np.nan; ys1[~np.isfinite(ys1)] = np.nan
                if len(fin): ys1 = np.clip(ys1, q1-4*spread, q3+4*spread)
                self.ax.plot(xs, ys1, color=CYAN, linewidth=1, linestyle="--",
                             alpha=0.5, label="f'(x)", zorder=3)
            except: pass

        # Roots
        for r in R["roots"]:
            rx = safe_float(r)
            if rx is not None and xmin <= rx <= xmax:
                self.ax.scatter([rx], [0], color=GREEN, s=60, zorder=6,
                                marker="o", edgecolors="white", linewidths=0.8)
                self.ax.annotate(f" x={sp.nsimplify(r,rational=True)}", xy=(rx,0),
                                 fontsize=7, color=GREEN, xytext=(2,6), textcoords="offset points")

        # Critical points
        for cp, fv, kind in R["classified"]:
            cx = safe_float(cp); cy = safe_float(fv)
            if cx is not None and cy is not None and xmin <= cx <= xmax:
                col = GREEN if "MIN" in kind else RED
                self.ax.scatter([cx], [cy], color=col, s=90, zorder=7,
                                edgecolors="white", linewidths=0.8)
                self.ax.annotate(f" {'min' if 'MIN' in kind else 'max'}\n ({cx:.2f},{cy:.2f})",
                                 xy=(cx,cy), fontsize=7, color=col,
                                 xytext=(6,4), textcoords="offset points")

        # Inflection points
        for ip in R["inflection"]:
            ix = safe_float(ip)
            if ix is not None and xmin <= ix <= xmax:
                try:
                    iy = float(expr.subs(x, ip))
                    self.ax.scatter([ix], [iy], color=ORANGE, s=55, zorder=6,
                                    marker="D", edgecolors="white", linewidths=0.7)
                except: pass

        # Vertical asymptotes
        for e in R["excluded"]:
            if xmin <= e <= xmax:
                self.ax.axvline(e, color=RED, linewidth=1.2, linestyle=":", alpha=0.7, zorder=1)
                self.ax.text(e, self.ax.get_ylim()[1], f" x={sp.nsimplify(e,rational=True)}",
                             color=RED, fontsize=7, va='top')

        # Horizontal asymptotes
        for ha in R["horiz_asymp"]:
            try:
                val = float(ha.split("=")[1].split()[0])
                self.ax.axhline(val, color=YELLOW, linewidth=0.9, linestyle="--",
                                alpha=0.5, zorder=1)
            except: pass

        # Sign shading: green where f>0, red where f<0
        try:
            pos = np.where(ys > 0, ys, np.nan)
            neg = np.where(ys < 0, ys, np.nan)
            self.ax.fill_between(xs, 0, pos, alpha=0.06, color=GREEN, zorder=2)
            self.ax.fill_between(xs, 0, neg, alpha=0.06, color=RED, zorder=2)
        except: pass

        self.ax.set_title(f"f(x) = {expr}", color=FG, fontsize=10, pad=8)
        self.ax.legend(loc="best", fontsize=8, facecolor=PANEL,
                       edgecolor=ACCENT, labelcolor=FG)

        # Rebuild annotation
        self._annot = self.ax.annotate("", xy=(0,0), xytext=(14,14),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="#0e0e1a", ec=ACCENT2, lw=1.2),
            arrowprops=dict(arrowstyle="->", color=ACCENT2),
            color=FG, fontsize=9, visible=False)

        self.fig.tight_layout(pad=1.5)

        # Apply manual Y limits if auto_y is off
        if not self._auto_y.get():
            try:
                ylo = float(self.ymin.get())
                yhi = float(self.ymax.get())
                if ylo < yhi:
                    self.ax.set_ylim(ylo, yhi)
            except:
                pass

        self.canvas.draw()

    def _apply_ylim(self):
        """Apply manual y limits immediately without re-analyzing."""
        try:
            ylo = float(self.ymin.get())
            yhi = float(self.ymax.get())
            if ylo >= yhi:
                return
            self._auto_y.set(False)
            self._auto_y_btn.config(text="Auto Y ✗", fg=ORANGE)
            self.ax.set_ylim(ylo, yhi)
            self.canvas.draw()
        except:
            pass

    def _toggle_auto_y(self):
        if self._auto_y.get():
            # Switch to manual — keep current limits
            self._auto_y.set(False)
            self._auto_y_btn.config(text="Auto Y ✗", fg=ORANGE)
            ylo, yhi = self.ax.get_ylim()
            self.ymin.delete(0, tk.END); self.ymin.insert(0, f"{ylo:.2f}")
            self.ymax.delete(0, tk.END); self.ymax.insert(0, f"{yhi:.2f}")
        else:
            # Switch back to auto — re-plot
            self._auto_y.set(True)
            self._auto_y_btn.config(text="Auto Y ✓", fg=CYAN)
            if self._R:
                try:
                    xmin = float(self.xmin.get())
                    xmax = float(self.xmax.get())
                    self._plot(self._R, xmin, xmax)
                    ylo, yhi = self.ax.get_ylim()
                    self.ymin.delete(0, tk.END); self.ymin.insert(0, f"{ylo:.2f}")
                    self.ymax.delete(0, tk.END); self.ymax.insert(0, f"{yhi:.2f}")
                except:
                    pass

    def _hover(self, event):
        if event.inaxes != self.ax or self._xs is None:
            self._annot.set_visible(False); self.canvas.draw_idle(); return
        idx = np.searchsorted(self._xs, event.xdata)
        idx = np.clip(idx, 0, len(self._xs)-1)
        xv = self._xs[idx]; yv = self._ys[idx]
        if not np.isfinite(yv):
            self._annot.set_visible(False); self.canvas.draw_idle(); return
        self._annot.xy = (xv, yv)

        # Compute f' and f'' at hover point
        extra = ""
        if self._R and self._R["f1"] is not None:
            try:
                x = self._R["x"]
                d1 = float(self._R["f1"].subs(x, xv))
                extra += f"\nf'  = {d1:.4f}"
            except: pass
        if self._R and self._R["f2"] is not None:
            try:
                x = self._R["x"]
                d2 = float(self._R["f2"].subs(x, xv))
                extra += f"\nf'' = {d2:.4f}"
            except: pass

        self._annot.set_text(f"x = {xv:.4f}\ny = {yv:.4f}{extra}")
        self._annot.set_visible(True)
        self.canvas.draw_idle()

if __name__ == "__main__":
    App().mainloop()