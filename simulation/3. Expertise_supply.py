"""
Autor & Thompson (2025) "Expertise" with a non-uniform worker distribution.

Supply-side modification (demand-side left intact):
  Let f(phi) >= 0 be the density of workers at expertise level phi, with
      int_0^1 f(phi) dphi = L_bar.
  Baseline: f(phi) = L_bar (uniform).

Modified equilibrium conditions (only (14) and (15) change; (16), (17) keep
the same aggregate structure, but w(phi) picks up f(phi) through (15')):

  (14')  F(I) / Y       = int_0^I  (1-a) wg^{(1-a)(1-lam)-1} (r/eta)^{a(1-lam)} dphi
                          where   F(I) = int_0^I f(psi) dpsi

  (15')  f(phi) / Y     = (1-a) w(phi)^{(1-a)(1-lam)-1} (r/eta)^{a(1-lam)}  (phi > I)
         -> w(phi) = [ (1-a) (r/eta)^{a(1-lam)} Y / f(phi) ]^(1/beta),
            beta = 1 + (1-a)(lam-1).

  (16)   Kbar / Y       = int_0^I (a/eta) wg^{(1-a)(1-lam)} (r/eta)^{a(1-lam)-1} dphi
                        + int_I^1 (a/eta) w(phi)^{(1-a)(1-lam)} (r/eta)^{a(1-lam)-1} dphi

  (17)   1              = int_0^I wg^{(1-a)(1-lam)} (r/eta)^{a(1-lam)} dphi
                        + int_I^1 w(phi)^{(1-a)(1-lam)} (r/eta)^{a(1-lam)} dphi

Tested densities (all integrate to L_bar on [0,1]):
    Uniform       f = L_bar
    Linear up     f = L_bar (1 - s + 2 s phi)         (s in [0,1])
    Linear down   f = L_bar (1 + s - 2 s phi)         (s in [0,1])
    Hump-shaped   f = L_bar (1 - b cos(2 pi phi))     (b in [0,1))

Also preserves the repo-style nesting: the outer solver finds (r, w_g), while
Y is recovered inside from the implied equilibrium conditions. Each expert type
phi > I supplies density f(phi) in her own occupation. If f(phi) is large enough, it can
drive w(phi) below w_g, which violates Prop 1 -- the script prints a
diagnostic warning when that happens.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

from params import Lbar, Kbar, lam, theta, eta, phi_list, I_grid

# ------------------------------------------------------------------
# User-chosen tilt strength for the linear and hump-shaped densities.
# ------------------------------------------------------------------
s   = 0.5    # s:  0 = uniform, 1 = f(phi) = 2 phi (or 2(1-phi))
b   = 0.6    # b:  0 = uniform, 0.6 => middle peak 1.6 at phi = 0.5


# ------------------------------------------------------------------
# Worker-density functions  f(phi)  and their integrals  F(I).
# All densities normalized so that int_0^1 f dphi = L_bar.
# ------------------------------------------------------------------
def f_uniform(phi):
    phi = np.asarray(phi, dtype=float)
    return np.full_like(phi, Lbar)

def F_uniform(I):
    return Lbar * I


def f_linearup(phi):
    """Increasing linear density: more high-expertise workers."""
    phi = np.asarray(phi, dtype=float)
    return Lbar * (1.0 - s + 2.0 * s * phi)

def F_linearup(I):
    return Lbar * ((1.0 - s) * I + s * I * I)


def f_lineardown(phi):
    """Decreasing linear density: more low-expertise workers."""
    phi = np.asarray(phi, dtype=float)
    return Lbar * (1.0 + s - 2.0 * s * phi)

def F_lineardown(I):
    return Lbar * ((1.0 + s) * I - s * I * I)


def f_hump(phi):
    """Hump-shaped density with a single interior peak at phi = 0.5."""
    phi = np.asarray(phi, dtype=float)
    return Lbar * (1.0 - b * np.cos(2.0 * np.pi * phi))


def F_hump(I):
    return Lbar * (I - b / (2.0 * np.pi) * np.sin(2.0 * np.pi * I))


# ------------------------------------------------------------------
# Gauss-Legendre nodes
# ------------------------------------------------------------------
_GL_N = 96
_GL_U, _GL_W = np.polynomial.legendre.leggauss(_GL_N)


def gl_integrate(f, a, b):
    """Gauss-Legendre integration on [a, b]; `f` may be scalar- or array-valued."""
    if b <= a:
        return 0.0
    half = 0.5 * (b - a)
    mid  = 0.5 * (b + a)
    x = half * _GL_U + mid
    y = np.asarray(f(x), dtype=float)
    return half * np.dot(_GL_W, y)


# ------------------------------------------------------------------
# Model basics
# ------------------------------------------------------------------
def alpha(phi, I):
    """a(phi) = min(phi, I) / [theta*(1-phi) + phi]."""
    phi = np.asarray(phi)
    return np.minimum(phi, I) / (theta * (1.0 - phi) + phi)


def p_from_w(a, w, r):
    """Occupation unit cost p(phi) = w^(1-a) (r/eta)^a."""
    return w ** (1.0 - a) * (r / eta) ** a


def w_of_phi(phi, Y, r, I, f_dens, f_floor=1e-12):
    """Expert wage from (15'):
        w(phi) = [ (1-a) (r/eta)^{a(1-lam)} Y / f(phi) ]^(1/beta)
    where beta = 1 + (1-a)(lam - 1).  Scalar- and array-safe.
    """
    a     = alpha(phi, I)
    beta  = 1.0 + (1.0 - a) * (lam - 1.0)
    f_val = np.maximum(np.asarray(f_dens(phi), dtype=float), f_floor)
    base  = (1.0 - a) * (r / eta) ** (a * (1.0 - lam)) * Y / f_val
    w     = base ** (1.0 / beta)
    return float(w) if np.ndim(w) == 0 else w


def implied_Y_from_labor(F_of_I, int14):
    """From equation (14'): F(I) / Y = int14."""
    if int14 <= 0:
        raise RuntimeError(f"int14 must be positive, got {int14}")
    return F_of_I / int14


def implied_Y_from_capital(int16_total):
    """From equation (16): Kbar / Y = int16_total."""
    if int16_total <= 0:
        raise RuntimeError(f"int16_total must be positive, got {int16_total}")
    return Kbar / int16_total


def fixed_point_Y(I, r, wg, f_dens, Y_init, tol=1e-10, max_iter=500):
    """Given (I, r, wg), recover Y by iterating on equation (16)."""
    Y = float(Y_init)

    def cap_inexpert(phi):
        a = alpha(phi, I)
        p = p_from_w(a, wg, r)
        return (a / r) * p ** (1.0 - lam)

    int16_g = gl_integrate(cap_inexpert, 0.0, I)

    for _ in range(max_iter):
        def cap_expert(phi):
            a = alpha(phi, I)
            w = w_of_phi(phi, Y, r, I, f_dens)
            p = p_from_w(a, w, r)
            return (a / r) * p ** (1.0 - lam)

        def price_expert(phi):
            a = alpha(phi, I)
            w = w_of_phi(phi, Y, r, I, f_dens)
            p = p_from_w(a, w, r)
            return p ** (1.0 - lam)

        int16_e = gl_integrate(cap_expert, I, 1.0)
        int17_e = gl_integrate(price_expert, I, 1.0)

        Y_new = implied_Y_from_capital(int16_g + int16_e)
        if not np.isfinite(Y_new) or Y_new <= 0:
            raise RuntimeError(f"Non-finite or non-positive Y encountered at I={I:.3f}")

        if abs(Y_new - Y) < tol:
            return Y_new, int16_e, int17_e

        Y = Y_new

    raise RuntimeError(f"Y fixed-point did not converge at I={I:.3f}")


# ------------------------------------------------------------------
# Residuals of the outer system in (r, w_g); Y is recovered inside.
# ------------------------------------------------------------------
def residuals_rw(x, I, f_dens, F_of_I):
    """Solve for x = [r, wg], eliminating Y."""
    if not (0.0 < I < 1.0):
        raise ValueError(f"I must be in (0, 1), got {I}")

    r, wg = x
    if not np.isfinite([r, wg]).all() or r <= 0 or wg <= 0:
        return np.array([1e10, 1e10])

    def int14_integrand(phi):
        a = alpha(phi, I)
        return (1.0 - a) * wg ** ((1.0 - a) * (1.0 - lam) - 1.0) \
                         * (r / eta) ** (a * (1.0 - lam))

    def cap_inexpert(phi):
        a = alpha(phi, I)
        p = p_from_w(a, wg, r)
        return (a / r) * p ** (1.0 - lam)

    def price_inexpert(phi):
        a = alpha(phi, I)
        p = p_from_w(a, wg, r)
        return p ** (1.0 - lam)

    int14 = gl_integrate(int14_integrand, 0.0, I)
    int16_g = gl_integrate(cap_inexpert, 0.0, I)
    int17_g = gl_integrate(price_inexpert, 0.0, I)

    Y_from_14 = implied_Y_from_labor(F_of_I, int14)
    Y, int16_e, int17_e = fixed_point_Y(I, r, wg, f_dens, Y_init=Y_from_14)
    Y_from_16 = implied_Y_from_capital(int16_g + int16_e)

    eqY = Y_from_14 - Y_from_16
    eq17 = 1.0 - (int17_g + int17_e)

    res = np.array([eqY, eq17])
    if not np.isfinite(res).all():
        return np.array([1e10, 1e10])
    return res


def recover_Y(I, r, wg, f_dens, F_of_I):
    """After solving for (r, wg), recover Y."""
    def int14_integrand(phi):
        a = alpha(phi, I)
        return (1.0 - a) * wg ** ((1.0 - a) * (1.0 - lam) - 1.0) \
                         * (r / eta) ** (a * (1.0 - lam))

    int14 = gl_integrate(int14_integrand, 0.0, I)
    Y0 = implied_Y_from_labor(F_of_I, int14)
    Y, _, _ = fixed_point_Y(I, r, wg, f_dens, Y_init=Y0)
    return Y


def solve_equilibrium(I, guess_rw, f_dens, F_func):
    F_of_I = float(F_func(I))

    sol = root(lambda x: residuals_rw(x, I, f_dens, F_of_I),
               x0=np.array(guess_rw), method="hybr")

    if (not sol.success) or (not np.isfinite(sol.x).all()):
        raise RuntimeError(f"Root finder failed at I={I:.3f}: {sol.message}")

    if np.linalg.norm(sol.fun) > 1e-8:
        raise RuntimeError(
            f"Root finder returned an inaccurate solution at I={I:.3f}. "
            f"Residual norm = {np.linalg.norm(sol.fun):.3e}"
        )

    r, wg = sol.x
    Y = recover_Y(I, r, wg, f_dens, F_of_I)
    return Y, r, wg


# ------------------------------------------------------------------
# Labor allocation:
#   phi > I  -> L(phi) = f(phi)                           (supply density)
#   phi <= I -> L(phi) = (1-a)/wg * p(phi)^(1-lam) * Y   (same as baseline)
# ------------------------------------------------------------------
def labor_phi(phi, I, Y, r, wg, f_dens):
    a = alpha(phi, I)
    if phi > I:
        return float(f_dens(phi))
    p = p_from_w(a, wg, r)
    return (1.0 - a) / wg * p ** (1.0 - lam) * Y


# ------------------------------------------------------------------
# Main run
# ------------------------------------------------------------------
def run(f_dens, F_func, grid=None, phi_targets=None, label=None,
        track_time=False):
    if grid is None:
        grid = I_grid
    if phi_targets is None:
        phi_targets = phi_list

    run_start = time.perf_counter()

    Y_vals, r_vals, wg_vals = [], [], []
    w_phi = {p: [] for p in phi_targets}
    L_phi = {p: [] for p in phi_targets}
    prop1_violations = []

    guess_rw = np.array([0.13, 1.29])
    for I in grid:
        Y, r, wg = solve_equilibrium(I, guess_rw, f_dens, F_func)
        guess_rw = np.array([r, wg])
        Y_vals.append(Y); r_vals.append(r); wg_vals.append(wg)

        for p in phi_targets:
            if p > I:
                w_p = w_of_phi(p, Y, r, I, f_dens)
                w_phi[p].append(float(w_p))
                L_phi[p].append(float(f_dens(p)))
                if w_p < wg:
                    prop1_violations.append((I, p, float(w_p), wg))
            else:
                w_phi[p].append(wg)
                L_phi[p].append(labor_phi(p, I, Y, r, wg, f_dens))

    total_time = time.perf_counter() - run_start
    if track_time:
        run_name = label if label is not None else getattr(f_dens, "__name__", "run")
        print(f"  {run_name}: {total_time:.3f}s")

    if prop1_violations:
        print(f"  WARN [{label}]: Prop 1 violated at {len(prop1_violations)} "
              f"(I, phi) points -- w(phi) < wg.  First: {prop1_violations[0]}")

    return {
        "I":     np.asarray(grid),
        "Y":     np.asarray(Y_vals),
        "r":     np.asarray(r_vals),
        "wg":    np.asarray(wg_vals),
        "w_phi": {p: np.asarray(v) for p, v in w_phi.items()},
        "L_phi": {p: np.asarray(v) for p, v in L_phi.items()},
        "time":  total_time,
    }


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
BLUE  = {0.8: "#08306b", 0.6: "#2171b5", 0.4: "#6baed6", 0.2: "#c6dbef"}
GREEN = {0.8: "#00441b", 0.6: "#238b45", 0.4: "#66c2a4", 0.2: "#ccece6"}


def plot_density(f_dens, ax, title):
    phi = np.linspace(0.0, 1.0, 401)
    ax.plot(phi, f_dens(phi), color="#5a4a9f", lw=2)
    ax.axhline(Lbar, color="grey", lw=1, linestyle="--", alpha=0.5,
               label="uniform $L_{bar}$")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$f(\phi)$")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", frameon=False, fontsize=9)


def plot_fig2(res, ax, title):
    I, r, wg = res["I"], res["r"], res["wg"]
    w_phi = res["w_phi"]
    for phi in [0.8, 0.6, 0.4, 0.2]:
        col  = BLUE[phi]
        pre  = I < phi
        post = I >= phi
        ax.plot(I[pre],  w_phi[phi][pre],  color=col, lw=2, label=fr"$w({phi})$")
        if post.any():
            ax.plot(I[post], w_phi[phi][post], color=col, lw=2)
        idx_hi = np.searchsorted(I, phi) - 1
        idx_lo = np.searchsorted(I, phi)
        if 0 <= idx_hi < len(I) and 0 <= idx_lo < len(I):
            ax.annotate("", xy=(phi, w_phi[phi][idx_lo]),
                        xytext=(phi, w_phi[phi][idx_hi]),
                        arrowprops=dict(arrowstyle="->", linestyle="dashed",
                                        color=col, lw=1.2))
    ax.plot(I, wg, color="#2ca25f", lw=2, label=r"$w_g$")
    ax.plot(I, r,  color="#e3455a", lw=2, label=r"$r$")
    ax.set_xlabel(r"$I$")
    ax.set_xlim(0.08, 0.72)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=9)


def plot_fig3(res, ax, title, f_dens):
    """Modified Fig 3: the 'phi > I' line is no longer constant -- it
    traces out f(phi) itself.  We show both f(phi) (dashed) and the
    generic L(phi) paths."""
    I = res["I"]
    L_phi = res["L_phi"]

    # reference curve: f(phi) sampled on phi = I.  Since phi > I means
    # L(phi) = f(phi) at that phi, we instead plot f(phi) on phi in [0,1].
    phi_grid = np.linspace(0.0, 1.0, 401)
    ax.plot(phi_grid, f_dens(phi_grid), color="#3A5FA8", lw=1.4,
            linestyle=":", label=r"$f(\phi)$  (expert supply)")

    for phi in [0.8, 0.6, 0.4, 0.2]:
        mask = I > phi
        if mask.any():
            ax.plot(I[mask], L_phi[phi][mask], color=GREEN[phi], lw=2,
                    label=fr"$\phi = {phi}$")
        # dashed arrow from L = f(phi) (pre-automation) to post jump
        idx = int(np.searchsorted(I, phi))
        if 0 < idx < len(I):
            ax.annotate("", xy=(I[idx], L_phi[phi][idx]),
                        xytext=(I[idx], float(f_dens(phi))),
                        arrowprops=dict(arrowstyle="->", linestyle="dashed",
                                        color=GREEN[phi], lw=1.2))
    ax.set_xlabel(r"$I$")
    ax.set_ylabel(r"$L(\phi)$")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=8)


# ------------------------------------------------------------------
# Dense grid for L(phi) vs phi figure
# ------------------------------------------------------------------
_PHI_DENSE = np.linspace(0.001, 0.999, 400)
_I_LEVELS  = [0.10, 0.20, 0.35, 0.50, 0.65]


def plot_fig4(res, f_dens, ax, title):
    """L(phi) vs phi for selected I levels."""
    I_arr = res["I"]
    colors = [plt.cm.plasma(v) for v in np.linspace(0.15, 0.85, len(_I_LEVELS))]
    for I_val, col in zip(_I_LEVELS, colors):
        idx  = int(np.argmin(np.abs(I_arr - I_val)))
        I_eq = I_arr[idx]
        Y, r, wg = res["Y"][idx], res["r"][idx], res["wg"][idx]
        L_vals = np.array([labor_phi(phi, I_eq, Y, r, wg, f_dens) for phi in _PHI_DENSE])
        ax.plot(_PHI_DENSE, L_vals, color=col, lw=1.5, label=fr"$I={I_eq:.2f}$")
        ax.axvline(I_eq, color=col, lw=0.8, linestyle=":")
    ax.axhline(Lbar, color="grey", lw=1.2, linestyle="--", label=r"$\bar{L}$")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$L(\phi)$")
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False, fontsize=8)


# ------------------------------------------------------------------
# Variants to run
# ------------------------------------------------------------------
VARIANTS = [
    (f_uniform,    F_uniform,    "Uniform  $f(\\phi)=L_{bar}$"),
    (f_linearup,   F_linearup,   rf"Linear up  $f(\phi)=L_{{bar}}(1-{s:.2f}+{2*s:.2f}\phi)$"),
    (f_lineardown, F_lineardown, rf"Linear down  $f(\phi)=L_{{bar}}(1+{s:.2f}-{2*s:.2f}\phi)$"),
    (f_hump,       F_hump,       rf"Hump-shaped  $f(\phi)=L_{{bar}}(1-{b:.2f}\cos 2\pi\phi)$"),
]


if __name__ == "__main__":
    script_start = time.perf_counter()
    results = []
    print("Solving supply-side variants...")
    for f_dens, F_func, label in VARIANTS:
        res = run(f_dens, F_func, label=label, track_time=True)
        results.append((res, label, f_dens))

    # --- Densities (one row, 4 panels) ------------------------------
    fig_d, axes_d = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
    for ax, (_, label, f_dens) in zip(axes_d, results):
        plot_density(f_dens, ax, label)
    fig_d.suptitle(r"Worker-expertise density $f(\phi)$", fontsize=13)
    fig_d.tight_layout()

    # --- Figure 2: 2x2 grid ----------------------------------------
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    for ax, (res, label, _) in zip(axes2.flat, results):
        plot_fig2(res, ax, label)
    fig2.suptitle(r"Figure 2 analog: $w(\phi),\ w_g,\ r$ vs $I$  (supply-side variants)",
                  fontsize=13)
    fig2.tight_layout()

    # --- Figure 3: 2x2 grid ----------------------------------------
    fig3, axes3 = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    for ax, (res, label, f_dens) in zip(axes3.flat, results):
        plot_fig3(res, ax, label, f_dens)
    fig3.suptitle(r"Figure 3 analog: $L(\phi)$ vs $I$  (supply-side variants)",
                  fontsize=13)
    fig3.tight_layout()

    # --- Figure 4: 2x2 grid ----------------------------------------
    fig4, axes4 = plt.subplots(2, 2, figsize=(15, 10), sharey=False)
    for ax, (res, label, f_dens) in zip(axes4.flat, results):
        plot_fig4(res, f_dens, ax, label)
    fig4.suptitle(r"Figure 4: $L(\phi)$ vs $\phi$  (supply-side variants)",
                  fontsize=13)
    fig4.tight_layout()

    total_script_time = time.perf_counter() - script_start
    print(f"\nOverall script time: {total_script_time:.3f} sec")

    plt.show()