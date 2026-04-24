import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch  # for arrows in plots
from scipy.integrate import quad_vec
from scipy.optimize import root
from params import Lbar, Kbar, lam, theta, eta, phi_list, I_grid, QUAD_LIMIT


# -----------------------------
# Model objects
# -----------------------------
# Page 12 defines I-share alpha(phi, I)
def alpha(phi, I):
    """Automation share alpha(phi, I) = min(phi, I) / [theta(1-phi) + phi]."""
    return min(phi, I) / (theta * (1.0 - phi) + phi)


# Equation (15): w(phi)
def w_expert(phi, I, Y, r, I_share=None):
    """
    Recover w(phi) for expert occupations phi > I from equation (15).
    Pass I_share to avoid recomputing alpha(phi, I) if already known.
    """
    if phi < I:
        raise ValueError("w_expert is only valid for phi >= I")

    if I_share is None:
        I_share = alpha(phi, I)

    exponent = 1.0 + (1.0 - I_share) * (lam - 1.0)

    return (
        (1.0 - I_share)
        * (r / eta) ** (I_share * (1.0 - lam))
        * (Y / Lbar)
    ) ** (1.0 / exponent)


# Equation (11): L(phi)
def labor_phi(phi, I, Y, r, wg, I_share=None):
    """
    Labor employed in occupation phi.
    - If phi > I (expert occupation): Proposition 1 implies L(phi) = Lbar.
    - If phi <= I (inexpert occupation): recover L(phi) from equation (11).
    """
    if I_share is None:
        I_share = alpha(phi, I)

    if phi > I:
        return Lbar

    return (
        Y
        * (1.0 - I_share)
        / wg
        * (wg ** (1.0 - I_share) * (r / eta) ** I_share) ** (1.0 - lam)
    )


# -----------------------------
# Integrands in Equations (14), (16), (17)
# -----------------------------
def integrand_generic(phi, I, wg, r):
    """
    Returns [eq14, eq16_generic, eq17_generic] integrands at phi, integrating over [0, I].
    """
    a     = alpha(phi, I)
    exp_w = (1.0 - a) * (1.0 - lam)
    exp_r = a * (1.0 - lam)

    f14 = (1.0 - a) * wg ** (exp_w - 1.0) * (r / eta) ** exp_r
    f16 = (a / eta)  * wg ** exp_w         * (r / eta) ** (exp_r - 1.0)
    f17 =              wg ** exp_w         * (r / eta) ** exp_r

    return np.array([f14, f16, f17])


def integrand_expert(phi, I, Y, r):
    """
    Returns [eq16_expert, eq17_expert] integrands at phi, integrating over [I, 1].
    """
    a     = alpha(phi, I)
    w     = w_expert(phi, I, Y, r, I_share=a)
    exp_w = (1.0 - a) * (1.0 - lam)
    exp_r = a * (1.0 - lam)

    f16 = (a / eta) * w ** exp_w * (r / eta) ** (exp_r - 1.0)
    f17 =             w ** exp_w * (r / eta) ** exp_r

    return np.array([f16, f17])


# -----------------------------
# Implied Y from equations (14) and (16)
# -----------------------------
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


# -----------------------------
# Inner fixed point for Y, given (r, wg, I)
# -----------------------------
def fixed_point_Y(I, r, wg, Y_init, tol=1e-10, max_iter=500):
    """
    Given (I, r, wg), recover Y by iterating on equation (16):
        Y_{new} = Kbar / [int16_g + int16_e(Y_old)].
    """
    Y = float(Y_init)

    # int16_g does not depend on Y – compute once outside the loop
    g_res, _ = quad_vec(lambda phi: integrand_generic(phi, I, wg, r), 0.0, I, limit=QUAD_LIMIT)
    if not np.isfinite(g_res).all():
        raise RuntimeError(f"Generic integral returned non-finite value at I={I:.3f}")
    _, int16_g, _ = g_res

    for iteration in range(max_iter):
        e_res, _ = quad_vec(lambda phi: integrand_expert(phi, I, Y, r), I, 1.0, limit=QUAD_LIMIT)
        if not np.isfinite(e_res).all():
            raise RuntimeError(f"Expert integral returned non-finite value at I={I:.3f}")
        int16_e, int17_e = e_res

        Y_new = implied_Y_from_capital(int16_g + int16_e)

        if not np.isfinite(Y_new) or Y_new <= 0:
            raise RuntimeError(f"Non-finite or non-positive Y encountered at I={I:.3f}")

        if abs(Y_new - Y) < tol:
            # return converged expert integrals so callers don't need to recompute
            return Y_new, int16_e, int17_e

        Y = Y_new

    raise RuntimeError(f"Y fixed-point did not converge at I={I:.3f}")


# -----------------------------
# GE system in (r, wg)
# -----------------------------
def residuals_rw(x, I):
    """
    Solve for x = [r, wg], eliminating Y.
    Residuals:
      1) Y implied by (14) minus Y implied by (16)
      2) equation (17)
    """
    if not (0.0 < I < 1.0):
        raise ValueError(f"I must be in (0, 1), got {I}")

    r, wg = x

    if not np.isfinite([r, wg]).all() or r <= 0 or wg <= 0:
        return np.array([1e10, 1e10])

    # Generic integrals do not depend on Y
    g, _ = quad_vec(lambda phi: integrand_generic(phi, I, wg, r), 0.0, I, limit=QUAD_LIMIT)
    if not np.isfinite(g).all():
        raise RuntimeError(f"Generic integral returned non-finite value at I={I:.3f}")
    int14, int16_g, int17_g = g

    # Y from equation (14)
    Y_from_14 = implied_Y_from_labor(I, int14)

    # Solve Y fixed point; expert integrals at convergence are returned directly
    Y, int16_e, int17_e = fixed_point_Y(I, r, wg, Y_init=Y_from_14)

    # Y from equation (16)
    Y_from_16 = implied_Y_from_capital(int16_g + int16_e)

    # Equation (17)
    eq17 = 1.0 - (int17_g + int17_e)

    # Residual 1: the two implied Y values must match
    eqY = Y_from_14 - Y_from_16

    return np.array([eqY, eq17])


def recover_Y(I, r, wg):
    """
    After solving for (r, wg), recover Y.
    """
    g, _ = quad_vec(lambda phi: integrand_generic(phi, I, wg, r), 0.0, I, limit=QUAD_LIMIT)
    if not np.isfinite(g).all():
        raise RuntimeError(f"Generic integral returned non-finite value at I={I:.3f}")
    int14, _, _ = g

    Y0 = implied_Y_from_labor(I, int14)
    Y, _, _ = fixed_point_Y(I, r, wg, Y_init=Y0)
    return Y


def solve_equilibrium(I, guess_rw):
    """
    Solve only for [r, wg], then recover Y endogenously.
    """
    sol = root(lambda x: residuals_rw(x, I), x0=np.array(guess_rw), method="hybr")

    if (not sol.success) or (not np.isfinite(sol.x).all()):
        raise RuntimeError(f"Root finder failed at I={I:.3f}: {sol.message}")

    if np.linalg.norm(sol.fun) > 1e-8:
        raise RuntimeError(
            f"Root finder returned an inaccurate solution at I={I:.3f}. "
            f"Residual norm = {np.linalg.norm(sol.fun):.3e}"
        )

    r, wg = sol.x
    Y = recover_Y(I, r, wg)

    return Y, r, wg


# -----------------------------
# Solve across a grid of I
# -----------------------------
results = {
    "I": [],
    "Y": [],
    "r": [],
    "wg": [],
    "w_by_phi": {phi: [] for phi in phi_list},
    "L_by_phi": {phi: [] for phi in phi_list},
}

# Only guess (r, wg); Y will be recovered endogenously
guess_rw = np.array([0.13, 1.29])

for I in I_grid:
    Y, r, wg = solve_equilibrium(I, guess_rw)
    guess_rw = np.array([r, wg])  # continuation in the reduced system

    results["I"].append(I)
    results["Y"].append(Y)
    results["r"].append(r)
    results["wg"].append(wg)

    for phi in phi_list:
        if phi <= I:
            wphi = wg
        else:
            wphi = w_expert(phi, I, Y, r)
        results["w_by_phi"][phi].append(wphi)

        Lphi = labor_phi(phi, I, Y, r, wg)
        results["L_by_phi"][phi].append(Lphi)


# -----------------------------
# Shared plot helpers
# -----------------------------
phi_colors = {0.2: "#A8D1E7", 0.4: "#5B9BBF", 0.6: "#1C5D8A", 0.8: "#0A2B4E"}
phi_lw     = {0.2: 1.2,       0.4: 1.2,       0.6: 1.4,       0.8: 1.6      }
I_arr      = np.array(results["I"], dtype=float)

def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="lightgray", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=11)
    plt.tight_layout()


# -----------------------------
# Plot Figure 2 replica
# -----------------------------
fig, ax = plt.subplots(figsize=(9, 6))

for phi in sorted(phi_list, reverse=True):
    x = I_arr
    y = np.array(results["w_by_phi"][phi], dtype=float)

    left = x < phi
    right = x > phi

    ax.plot(x[left], y[left],
            color=phi_colors[phi], linewidth=phi_lw[phi], label=fr"$w({phi})$")
    ax.plot(x[right], y[right],
            color=phi_colors[phi], linewidth=phi_lw[phi])

ax.plot(results["I"], results["wg"], color="#4A7C59", linewidth=1.5, label=r"$w_g$")
ax.plot(results["I"], results["r"],  color="#D97A7A", linewidth=1.5, label=r"$r$")

for phi in phi_list:
    idx = int(np.searchsorted(I_arr, phi)) - 1
    if 0 <= idx < len(I_arr):
        x     = I_arr[idx]
        y_top = results["w_by_phi"][phi][idx]
        y_bot = results["wg"][idx]
        arrow = FancyArrowPatch(
            posA=(x, y_top), posB=(x, y_bot),
            arrowstyle="-|>", color=phi_colors[phi],
            linewidth=1.4, linestyle="dashed",
            mutation_scale=8, transform=ax.transData,
        )
        ax.add_patch(arrow)

ax.set_xlabel(r"$I$", fontsize=13)
ax.set_xlim(0.08, 0.72)
ax.set_ylim(0, 5)
style_ax(ax)


# -----------------------------
# Plot Figure 3 replica
# -----------------------------
phi_colors3 = {0.2: "#A8D8A8", 0.4: "#5B9B5B", 0.6: "#1C5D1C", 0.8: "#0A2B0A"}

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(
    results["I"],
    [Lbar] * len(results["I"]),
    color="#3A5FA8",
    linewidth=1.8,
    linestyle="solid",
    label=r"$\phi > I$"
)

for phi in sorted(phi_list, reverse=True):
    mask = I_arr > phi
    if mask.any():
        ax.plot(
            I_arr[mask],
            np.array(results["L_by_phi"][phi])[mask],
            color=phi_colors3[phi],
            linewidth=phi_lw[phi],
            label=fr"$\phi = {phi}$"
        )

for phi in phi_list:
    idx = int(np.searchsorted(I_arr, phi))
    if 0 < idx < len(I_arr):
        x     = I_arr[idx]
        y_bot = Lbar
        y_top = results["L_by_phi"][phi][idx]
        arrow = FancyArrowPatch(
            posA=(x, y_bot),
            posB=(x, y_top),
            arrowstyle="-|>",
            color=phi_colors3[phi],
            linewidth=1.4,
            linestyle="dashed",
            mutation_scale=8,
            transform=ax.transData,
        )
        ax.add_patch(arrow)

ax.set_xlabel(r"$I$", fontsize=13)
ax.set_ylabel(r"$L(\phi)$", fontsize=13)
ax.set_xlim(0.08, 0.72)
ax.set_ylim(0, 3.5)
style_ax(ax)


# -----------------------------
# Plot Figure: L(phi) vs phi for selected I values
# -----------------------------
I_levels   = [0.10, 0.20, 0.35, 0.50, 0.65]
phi_dense  = np.linspace(0.001, 0.999, 400)
cmap4      = plt.cm.plasma
I_colors   = [cmap4(v) for v in np.linspace(0.15, 0.85, len(I_levels))]

I_arr_full = np.array(results["I"], dtype=float)
Y_arr      = np.array(results["Y"], dtype=float)
r_arr      = np.array(results["r"], dtype=float)
wg_arr     = np.array(results["wg"], dtype=float)

fig, ax = plt.subplots(figsize=(9, 6))

for I_val, col in zip(I_levels, I_colors):
    idx = int(np.argmin(np.abs(I_arr_full - I_val)))
    I_eq = I_arr_full[idx]
    Y_eq, r_eq, wg_eq = Y_arr[idx], r_arr[idx], wg_arr[idx]

    L_vals = np.array([labor_phi(phi, I_eq, Y_eq, r_eq, wg_eq)
                       for phi in phi_dense])
    ax.plot(phi_dense, L_vals, color=col, lw=1, label=fr"$I = {I_eq:.2f}$")
    ax.axvline(I_eq, color=col, lw=0.8, linestyle=":")

ax.axhline(Lbar, color="grey", lw=1.2, linestyle="--", label=r"$\bar{L}$")
ax.set_xlabel(r"$\phi$", fontsize=13)
ax.set_ylabel(r"$L(\phi)$", fontsize=13)
ax.set_xlim(0.0, 1.0)
style_ax(ax)

plt.show()