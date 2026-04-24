"""
Autor & Thompson (2025) "Expertise" with industry-specific weights in the
final-good aggregator.

    Y = [ int_0^1 omega(phi) * Y(phi)^((lam-1)/lam) dphi ]^(lam/(lam-1))

Baseline weight: omega(phi) = 1.

Effect on equilibrium: cost minimization gives
    Y(phi) = omega(phi)^lam * p(phi)^(-lam) * Y          (numeraire P = 1)
    L(phi) = (1-a) * omega(phi)^lam * w(phi)^((1-a)*(1-lam)-1) * Y
    K(phi) = (a/eta) * omega(phi)^lam * (w(phi)^((1-a)*(1-lam))) * (r/eta)^(a*(1-lam)-1) * Y
    w(phi)^(1+(1-a)*(lam-1)) = (1-a) * omega(phi)^lam * (r/eta)^(a*(1-lam)) * Y / L_bar

So every integrand and the expert-wage argument gain a factor omega(phi)^lam.
Proposition 1 (L(phi) = L_bar for phi > I) is preserved.

Here w(phi) = w_g on phi <= I, while for phi > I it is determined by the
expert wage equation above.

This version uses the same weighted model objects, but matches the repo-style
solver nesting: the outer solver finds (r, w_g), while Y is recovered inside
from the implied equilibrium conditions.
"""

import time
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

from params import Lbar, Kbar, lam, theta, eta, phi_list, I_grid

# ------------------------------------------------------------------
# User-chosen slope strength for linear weights.
# s = 0   -> uniform
# s = 0.5 -> moderate tilt
# s = 1   -> extreme tilt (old 2*phi or 2*(1-phi))
# ------------------------------------------------------------------
s = 0.5


# ------------------------------------------------------------------
# Industry weight function. Change OMEGA to try alternatives.
# ------------------------------------------------------------------
def omega_uniform(_phi):
    """Baseline uniform weights (reproduces Figure 2 and 3 exactly)."""
    return 1.0


def omega_linearup(phi):
    """Upward-sloping linear weight, normalized to integrate to 1 on [0,1]."""
    return 1.0 - s + 2.0 * s * phi


def omega_lineardown(phi):
    """Downward-sloping linear weight, normalized to integrate to 1 on [0,1]."""
    return 1.0 + s - 2.0 * s * phi


def omega_logcav(_phi):
    """Example of a log-concave weight function."""
    return 1.5 * (1.0 - _phi**2)


# -------------------------------------------------------------------------------------------
# Gauss-Legendre nodes
# -------------------------------------------------------------------------------------------
_GL_N = 96
_GL_U, _GL_W = np.polynomial.legendre.leggauss(_GL_N)


def gl_nodes_weights(a, b):
    """Return transformed Gauss-Legendre nodes and weights on [a,b]."""
    if b <= a:
        return np.empty(0), np.empty(0)
    half = 0.5 * (b - a)
    mid = 0.5 * (b + a)
    x = half * _GL_U + mid
    w = half * _GL_W
    return x, w


# ------------------------------------------------------------------
# Model basics
# ------------------------------------------------------------------
def alpha(phi, I):
    """Automation share alpha(phi, I) = min(phi, I) / [theta(1-phi) + phi]."""
    phi = np.asarray(phi)
    return np.minimum(phi, I) / (theta * (1.0 - phi) + phi)


def omega_lam(phi, omega):
    """omega(phi)^lam."""
    return omega(phi) ** lam


def weight_id(omega):
    """Stable identifier for caching per-weight integration data."""
    return getattr(omega, "__name__", str(id(omega)))


@lru_cache(maxsize=None)
def integration_cache(I, omega_name):
    """
    Precompute nodes, weights, alpha(phi,I), and omega(phi)^lam
    on both integration regions [0,I] and [I,1].
    """
    omega_map = {
        "omega_uniform": omega_uniform,
        "omega_linearup": omega_linearup,
        "omega_lineardown": omega_lineardown,
        "omega_logcav": omega_logcav,
    }
    omega_fn = omega_map[omega_name]

    phi_g, wgts_g = gl_nodes_weights(0.0, I)
    phi_e, wgts_e = gl_nodes_weights(I, 1.0)

    a_g = alpha(phi_g, I) if phi_g.size else np.empty(0)
    a_e = alpha(phi_e, I) if phi_e.size else np.empty(0)

    om_g = omega_lam(phi_g, omega_fn) if phi_g.size else np.empty(0)
    om_e = omega_lam(phi_e, omega_fn) if phi_e.size else np.empty(0)

    return {
        "phi_g": phi_g,
        "wgts_g": wgts_g,
        "a_g": a_g,
        "om_g": om_g,
        "phi_e": phi_e,
        "wgts_e": wgts_e,
        "a_e": a_e,
        "om_e": om_e,
    }


def w_expert(phi, Y, r, I, omega):
    """
    Expert wage from modified (15):
        w(phi) = [ (1-a) * omega(phi)^lam * (r/eta)^{a(1-lam)} * Y/L_bar ]^(1/beta)
    with beta = 1 + (1-a)(lam-1).
    """
    a = alpha(phi, I)
    beta = 1.0 + (1.0 - a) * (lam - 1.0)
    om_l = omega_lam(phi, omega)
    base = (1.0 - a) * om_l * (r / eta) ** (a * (1.0 - lam)) * Y / Lbar
    w = np.where(om_l <= 0.0, 0.0, base ** (1.0 / beta))
    return float(w) if np.ndim(w) == 0 else w


def implied_Y_from_labor(I, int14):
    """From equation (14): I * Lbar / Y = int14."""
    if int14 <= 0:
        raise RuntimeError(f"int14 must be positive, got {int14}")
    return I * Lbar / int14


def implied_Y_from_capital(int16_total):
    """From equation (16): Kbar / Y = int16_total."""
    if int16_total <= 0:
        raise RuntimeError(f"int16_total must be positive, got {int16_total}")
    return Kbar / int16_total


def fixed_point_Y(I, r, wg, omega, Y_init, tol=1e-10, max_iter=500):
    """Given (I, r, wg), recover Y by iterating on equation (16)."""
    Y = float(Y_init)
    cache = integration_cache(float(I), weight_id(omega))

    a_g = cache["a_g"]
    om_g = cache["om_g"]
    wgts_g = cache["wgts_g"]

    exp_w_g = (1.0 - a_g) * (1.0 - lam)
    exp_r_g = a_g * (1.0 - lam)
    int16_g_vals = om_g * (a_g / eta) * (wg ** exp_w_g) * (r / eta) ** (exp_r_g - 1.0)
    int16_g = float(np.dot(wgts_g, int16_g_vals)) if int16_g_vals.size else 0.0

    a_e = cache["a_e"]
    om_e = cache["om_e"]
    wgts_e = cache["wgts_e"]
    phi_e = cache["phi_e"]

    for _ in range(max_iter):
        if phi_e.size:
            w_e = w_expert(phi_e, Y, r, I, omega)
            exp_w_e = (1.0 - a_e) * (1.0 - lam)
            exp_r_e = a_e * (1.0 - lam)
            int16_e_vals = om_e * (a_e / eta) * (w_e ** exp_w_e) * (r / eta) ** (exp_r_e - 1.0)
            int17_e_vals = om_e * (w_e ** exp_w_e) * (r / eta) ** exp_r_e
            int16_e = float(np.dot(wgts_e, int16_e_vals))
            int17_e = float(np.dot(wgts_e, int17_e_vals))
        else:
            int16_e = 0.0
            int17_e = 0.0

        Y_new = implied_Y_from_capital(int16_g + int16_e)
        if not np.isfinite(Y_new) or Y_new <= 0:
            raise RuntimeError(f"Non-finite or non-positive Y encountered at I={I:.3f}")

        if abs(Y_new - Y) < tol:
            return Y_new, int16_e, int17_e

        Y = Y_new

    raise RuntimeError(f"Y fixed-point did not converge at I={I:.3f}")


# -------------------------------------------------------------------------
# Residuals of the outer system in (r, wg); Y is recovered inside.
# -------------------------------------------------------------------------
def residuals_rw(x, I, omega):
    """Solve for x = [r, wg], eliminating Y."""
    if not (0.0 < I < 1.0):
        raise ValueError(f"I must be in (0, 1), got {I}")

    r, wg = x
    if not np.isfinite([r, wg]).all() or r <= 0 or wg <= 0:
        return np.array([1e10, 1e10])

    cache = integration_cache(float(I), weight_id(omega))

    a_g = cache["a_g"]
    om_g = cache["om_g"]
    wgts_g = cache["wgts_g"]

    exp_w_g = (1.0 - a_g) * (1.0 - lam)
    exp_r_g = a_g * (1.0 - lam)

    int14_vals = om_g * (1.0 - a_g) * (wg ** (exp_w_g - 1.0)) * (r / eta) ** exp_r_g
    int16_g_vals = om_g * (a_g / eta) * (wg ** exp_w_g) * (r / eta) ** (exp_r_g - 1.0)
    int17_g_vals = om_g * (wg ** exp_w_g) * (r / eta) ** exp_r_g

    int14 = float(np.dot(wgts_g, int14_vals)) if int14_vals.size else 0.0
    int16_g = float(np.dot(wgts_g, int16_g_vals)) if int16_g_vals.size else 0.0
    int17_g = float(np.dot(wgts_g, int17_g_vals)) if int17_g_vals.size else 0.0

    Y_from_14 = implied_Y_from_labor(I, int14)
    Y, int16_e, int17_e = fixed_point_Y(I, r, wg, omega, Y_init=Y_from_14)
    Y_from_16 = implied_Y_from_capital(int16_g + int16_e)

    eqY = Y_from_14 - Y_from_16
    eq17 = 1.0 - (int17_g + int17_e)

    res = np.array([eqY, eq17])
    if not np.isfinite(res).all():
        return np.array([1e10, 1e10])
    return res


def recover_Y(I, r, wg, omega):
    """After solving for (r, wg), recover Y."""
    cache = integration_cache(float(I), weight_id(omega))
    a_g = cache["a_g"]
    om_g = cache["om_g"]
    wgts_g = cache["wgts_g"]

    exp_w_g = (1.0 - a_g) * (1.0 - lam)
    exp_r_g = a_g * (1.0 - lam)
    int14_vals = om_g * (1.0 - a_g) * (wg ** (exp_w_g - 1.0)) * (r / eta) ** exp_r_g
    int14 = float(np.dot(wgts_g, int14_vals)) if int14_vals.size else 0.0

    Y0 = implied_Y_from_labor(I, int14)
    Y, _, _ = fixed_point_Y(I, r, wg, omega, Y_init=Y0)
    return Y


def solve_equilibrium(I, guess_rw, omega):
    """Solve only for [r, wg], then recover Y endogenously."""
    sol = root(lambda x: residuals_rw(x, I, omega), x0=np.array(guess_rw), method="hybr")

    if (not sol.success) or (not np.isfinite(sol.x).all()):
        raise RuntimeError(f"Root finder failed at I={I:.3f}: {sol.message}")

    if np.linalg.norm(sol.fun) > 1e-8:
        raise RuntimeError(
            f"Root finder returned an inaccurate solution at I={I:.3f}. "
            f"Residual norm = {np.linalg.norm(sol.fun):.3e}"
        )

    r, wg = sol.x
    Y = recover_Y(I, r, wg, omega)
    return Y, r, wg


# ------------------------------------------------------------------
# L(phi):   phi > I  -> L_bar  (Prop 1)
#           phi <= I -> weighted labor demand
# ------------------------------------------------------------------
def labor_phi(phi, I, Y, r, wg, omega):
    a = alpha(phi, I)
    if phi > I:
        return Lbar

    exp_w = (1.0 - a) * (1.0 - lam)
    exp_r = a * (1.0 - lam)
    return omega_lam(phi, omega) * (1.0 - a) / wg * (wg ** exp_w) * ((r / eta) ** exp_r) * Y


# ------------------------------------------------------------------
# Main run
# ------------------------------------------------------------------
def run(omega, grid=None, phi_targets=None, label=None, track_time=False):
    if grid is None:
        grid = I_grid
    if phi_targets is None:
        phi_targets = phi_list

    run_start = time.perf_counter()

    Y_vals, r_vals, wg_vals = [], [], []
    w_phi = {p: [] for p in phi_targets}
    L_phi = {p: [] for p in phi_targets}

    guess_rw = np.array([0.13, 1.29])

    for I in grid:
        Y, r, wg = solve_equilibrium(I, guess_rw, omega)
        guess_rw = np.array([r, wg])

        Y_vals.append(Y)
        r_vals.append(r)
        wg_vals.append(wg)

        for p in phi_targets:
            w_phi[p].append(w_expert(p, Y, r, I, omega) if p > I else wg)
            L_phi[p].append(labor_phi(p, I, Y, r, wg, omega))

    total_time = time.perf_counter() - run_start
    if track_time:
        run_name = label if label is not None else getattr(omega, "__name__", "run")
        print(f"  Total time for {run_name}: {total_time:.4f} sec")

    return {
        "I": np.asarray(grid),
        "Y": np.asarray(Y_vals),
        "r": np.asarray(r_vals),
        "wg": np.asarray(wg_vals),
        "w_phi": {p: np.asarray(v) for p, v in w_phi.items()},
        "L_phi": {p: np.asarray(v) for p, v in L_phi.items()},
        "time": total_time,
    }


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
BLUE = {0.8: "#08306b", 0.6: "#2171b5", 0.4: "#6baed6", 0.2: "#c6dbef"}
GREEN = {0.8: "#00441b", 0.6: "#238b45", 0.4: "#66c2a4", 0.2: "#ccece6"}


def plot_fig2(res, ax, title):
    I, r, wg = res["I"], res["r"], res["wg"]
    w_phi = res["w_phi"]

    for phi in [0.8, 0.6, 0.4, 0.2]:
        col = BLUE[phi]
        pre = I < phi
        post = I >= phi
        ax.plot(I[pre], w_phi[phi][pre], color=col, lw=2, label=fr"$w({phi})$")
        if post.any():
            ax.plot(I[post], w_phi[phi][post], color=col, lw=2)

        idx_hi = np.searchsorted(I, phi) - 1
        idx_lo = np.searchsorted(I, phi)
        if 0 <= idx_hi < len(I) and 0 <= idx_lo < len(I):
            ax.annotate(
                "",
                xy=(phi, w_phi[phi][idx_lo]),
                xytext=(phi, w_phi[phi][idx_hi]),
                arrowprops=dict(arrowstyle="->", linestyle="dashed", color=col, lw=1.2),
            )

    ax.plot(I, wg, color="#2ca25f", lw=2, label=r"$w_g$")
    ax.plot(I, r, color="#e3455a", lw=2, label=r"$r$")
    ax.set_xlabel(r"$I$")
    ax.set_xlim(0.08, 0.72)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=9)


def plot_fig3(res, ax, title):
    I = res["I"]
    L_phi = res["L_phi"]

    ax.plot(I, np.full_like(I, Lbar), color="#3A5FA8", lw=2, label=r"$\phi > I$")

    for phi in [0.8, 0.6, 0.4, 0.2]:
        mask = I > phi
        if mask.any():
            ax.plot(I[mask], L_phi[phi][mask], color=GREEN[phi], lw=2, label=fr"$\phi = {phi}$")

        idx = int(np.searchsorted(I, phi))
        if 0 < idx < len(I):
            ax.annotate(
                "",
                xy=(I[idx], L_phi[phi][idx]),
                xytext=(I[idx], Lbar),
                arrowprops=dict(arrowstyle="->", linestyle="dashed", color=GREEN[phi], lw=1.2),
            )

    ax.set_xlabel(r"$I$")
    ax.set_ylabel(r"$L(\phi)$")
    ax.set_xlim(0.08, 0.72)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=9)


_PHI_DENSE = np.linspace(0.001, 0.999, 400)
_I_LEVELS = [0.10, 0.20, 0.35, 0.50, 0.65]


def plot_fig4(res, omega, ax, title):
    """L(phi) vs phi for several fixed I values, one curve per I."""
    I_grid_arr = res["I"]
    colors = [plt.cm.plasma(v) for v in np.linspace(0.15, 0.85, len(_I_LEVELS))]

    for I_val, col in zip(_I_LEVELS, colors):
        idx = int(np.argmin(np.abs(I_grid_arr - I_val)))
        I_eq = I_grid_arr[idx]
        Y, r, wg = res["Y"][idx], res["r"][idx], res["wg"][idx]

        L_vals = np.array([labor_phi(phi, I_eq, Y, r, wg, omega) for phi in _PHI_DENSE])
        ax.plot(_PHI_DENSE, L_vals, color=col, lw=1.5, label=fr"$I={I_eq:.2f}$")
        ax.axvline(I_eq, color=col, lw=0.8, linestyle=":")

    ax.axhline(Lbar, color="grey", lw=1.2, linestyle="--", label=r"$\bar{L}$")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$L(\phi)$")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False, fontsize=9)


# ------------------------------------------------------------------
VARIANTS = [
    (omega_uniform, r"Uniform  $\omega(\phi)=1$"),
    (omega_linearup, rf"Linear up  $\omega(\phi)={1-s:.2f}+{2*s:.2f}\phi$"),
    (omega_lineardown, rf"Linear down  $\omega(\phi)={1+s:.2f}-{2*s:.2f}\phi$"),
    (omega_logcav, r"Log-concave  $\omega(\phi)=1.5(1-\phi^2)$"),
]


if __name__ == "__main__":
    script_start = time.perf_counter()

    results = []
    for fn, label in VARIANTS:
        res = run(fn, label=label, track_time=True)
        results.append((res, label))

    # --- Figure 2 --------------------------------------------------
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    for ax, (res, label) in zip(axes2.flat, results):
        plot_fig2(res, ax, label)
    fig2.suptitle(
        r"Figure 2: wages $w(\phi)$, generic wage $w_g$, and rental rate $r$ vs $I$",
        fontsize=13,
    )
    fig2.tight_layout()

    # --- Figure 3 --------------------------------------------------
    fig3, axes3 = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    for ax, (res, label) in zip(axes3.flat, results):
        plot_fig3(res, ax, label)
    fig3.suptitle(
        r"Figure 3: labor allocation $L(\phi)$ vs automation cutoff $I$",
        fontsize=13,
    )
    fig3.tight_layout()

    # --- Figure 4 --------------------------------------------------
    fig4, axes4 = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
    for ax, ((res, label), (fn, _)) in zip(axes4.flat, zip(results, VARIANTS)):
        plot_fig4(res, fn, ax, label)
    fig4.suptitle(
        r"Figure 4: labor demand $L(\phi)$ as a function of $\phi$,"
        r" for selected automation cutoffs $I$",
        fontsize=13,
    )
    fig4.tight_layout()

    total_script_time = time.perf_counter() - script_start
    print(f"Overall script time: {total_script_time:.4f} sec")
    plt.show()
