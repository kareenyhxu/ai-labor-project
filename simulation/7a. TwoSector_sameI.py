"""
Expert-feasible downward-mobility PAV equilibrium.

Economic structure
------------------
Task is phi in [0, 1].
Technology threshold is I.

Generic / inexpert tasks:
    phi in [0, I]
    wage = w_g
    treated as a generic pool.

Expert tasks:
    phi in (I, 1]
    worker expertise e can perform any task phi <= e, but not phi > e.
    If an expert downgrades, she earns the task wage w(phi).

Implications:
    1. w_g remains an outside-loop/pooled/aggregated equilibrium variable.
    2. Expert task wages must be weakly increasing and at least w_g.
    3. If expert unconstrained wages violate monotonicity or fall below w_g,
       PAV creates flat blocks.
    4. Expert blocks must satisfy cumulative downward feasibility:

           int_z^b L(phi)dphi <= (b-z)Lbar

       inside each expert block [a,b].

Important distinction:
    The cumulative downward constraint is imposed on expert regions only.
    We do NOT impose expertise-ladder feasibility inside the generic pool [0,I].
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib_cache"),
)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, root

from params import Lbar, Kbar, lam, theta, eta


GRAPH_DIR = Path(__file__).parent / "sim_graphs"


# ------------------------------------------------------------------
# Numerical controls
# ------------------------------------------------------------------

EPS = 1e-12
DEFAULT_N_GRID = 500
ROOT_TOL = 1e-9
BLOCK_WAGE_TOL = 1e-11
PAV_TOL = 1e-11
FEAS_TOL = 1e-9
MAX_FEAS_REFINEMENTS = 200


# ------------------------------------------------------------------
# Demand Function Weights - Sector Heterogeneity
# ------------------------------------------------------------------

LINEAR_SLOPE = 0.5


def omega_up(phi, s=LINEAR_SLOPE):
    return 1.0 - s + 2.0 * s * phi


def omega_down(phi, s=LINEAR_SLOPE):
    return 1.0 + s - 2.0 * s * phi

# Aggregate weight, OM
def omega_components(phi, xi):
    phi = np.asarray(phi, dtype=float)

    om1_lam = omega_up(phi) ** lam
    om2_lam = omega_down(phi) ** lam
    OM = om1_lam + xi * om2_lam

    return om1_lam, om2_lam, OM


# ------------------------------------------------------------------
# Grid
# ------------------------------------------------------------------

def make_task_grid(I, n_grid=DEFAULT_N_GRID):
    """
    Finite-volume grid on [0,1], with I inserted as an exact edge.
    """
    base_edges = np.linspace(0.0, 1.0, n_grid + 1)
    edges = np.unique(np.sort(np.append(base_edges, I)))

    phi = 0.5 * (edges[:-1] + edges[1:])
    weights = np.diff(edges)

    n_anchor = int(np.searchsorted(edges, I, side="left"))

    if not np.isclose(edges[n_anchor], I):
        raise RuntimeError("Grid construction failed: I is not an exact edge.")

    return phi, weights, edges, n_anchor


# ------------------------------------------------------------------
# Model primitives
# ------------------------------------------------------------------

def alpha(phi, I):
    phi = np.asarray(phi, dtype=float)
    return np.minimum(phi, I) / (theta * (1.0 - phi) + phi)


def task_price(w, r, a):
    return w ** (1.0 - a) * (r / eta) ** a


def labor_demand(phi, w, r, Ytilde, xi, I):
    phi = np.asarray(phi, dtype=float)
    w = np.asarray(w, dtype=float)

    a = alpha(phi, I)
    _, _, OM = omega_components(phi, xi)

    p = task_price(w, r, a)
    p_power = p ** (1.0 - lam)

    return (1.0 - a) / w * OM * p_power * Ytilde


def capital_demand(phi, w, r, Ytilde, xi, I):
    phi = np.asarray(phi, dtype=float)
    w = np.asarray(w, dtype=float)

    a = alpha(phi, I)
    _, _, OM = omega_components(phi, xi)

    p = task_price(w, r, a)
    p_power = p ** (1.0 - lam)

    return (a / r) * OM * p_power * Ytilde


def unconstrained_expert_wage(phi, r, Ytilde, xi, I):
    """
    Expert unconstrained wage from L^d(phi; w)=Lbar.
    Used only for phi > I.
    """
    scalar_input = np.ndim(phi) == 0
    phi_arr = np.asarray(phi, dtype=float)

    a = alpha(phi_arr, I)
    _, _, OM = omega_components(phi_arr, xi)

    beta = 1.0 + (1.0 - a) * (lam - 1.0)

    base = (
        (1.0 - a)
        * OM
        * (r / eta) ** (a * (1.0 - lam))
        * Ytilde
        / Lbar
    )

    valid = (
        (OM > 0.0)
        & (base > 0.0)
        & np.isfinite(base)
        & np.isfinite(beta)
        & (np.abs(beta) > EPS)
    )

    out = np.full_like(phi_arr, np.nan, dtype=float)
    out[valid] = base[valid] ** (1.0 / beta[valid])

    if scalar_input:
        return float(out)

    return out


# ------------------------------------------------------------------
# Blocks
# ------------------------------------------------------------------

@dataclass
class Block:
    left: int
    right: int
    wage: float
    is_anchor: bool = False

    @property
    def size(self):
        return self.right - self.left


def block_supply(weights, left, right):
    return Lbar * float(np.sum(weights[left:right]))


def block_labor_demand_at_wage(phi, weights, left, right, wage, r, Ytilde, xi, I):
    L = labor_demand(phi[left:right], wage, r, Ytilde, xi, I)
    return float(np.dot(weights[left:right], L))


def solve_block_wage(phi, weights, left, right, r, Ytilde, xi, I):
    """
    Solve non-anchor block wage from:

        int_block L^d(phi; w_B)dphi = block supply.
    """
    target = block_supply(weights, left, right)

    def excess_demand(log_w):
        wage = np.exp(log_w)
        demand = block_labor_demand_at_wage(
            phi,
            weights,
            left,
            right,
            wage,
            r,
            Ytilde,
            xi,
            I,
        )
        return demand - target

    w_uc = unconstrained_expert_wage(phi[left:right], r, Ytilde, xi, I)
    finite = w_uc[np.isfinite(w_uc) & (w_uc > 0.0)]

    if finite.size == 0:
        raise RuntimeError(
            f"Cannot bracket non-anchor block wage for block [{left}, {right})."
        )

    low = np.log(max(np.nanmin(finite) * 1e-4, EPS))
    high = np.log(max(np.nanmax(finite) * 1e4, EPS * 10.0))

    f_low = excess_demand(low)
    f_high = excess_demand(high)

    expand_count = 0
    while f_low < 0.0 and expand_count < 80:
        low -= 2.0
        f_low = excess_demand(low)
        expand_count += 1

    expand_count = 0
    while f_high > 0.0 and expand_count < 80:
        high += 2.0
        f_high = excess_demand(high)
        expand_count += 1

    if not (np.isfinite(f_low) and np.isfinite(f_high)) or f_low * f_high > 0.0:
        raise RuntimeError(
            f"Failed to bracket block wage for block [{left}, {right}). "
            f"f_low={f_low}, f_high={f_high}"
        )

    log_w_star = brentq(excess_demand, low, high, xtol=BLOCK_WAGE_TOL)

    return float(np.exp(log_w_star))


def assign_wages_from_blocks(n, blocks):
    wages = np.empty(n, dtype=float)

    for block in blocks:
        wages[block.left:block.right] = block.wage

    return wages


def merge_two_blocks(left_block, right_block, phi, weights, wg, r, Ytilde, xi, I):
    merged_left = left_block.left
    merged_right = right_block.right
    is_anchor = left_block.is_anchor or right_block.is_anchor

    if is_anchor:
        merged_wage = wg
    else:
        merged_wage = solve_block_wage(
            phi,
            weights,
            merged_left,
            merged_right,
            r,
            Ytilde,
            xi,
            I,
        )

    return Block(
        left=merged_left,
        right=merged_right,
        wage=merged_wage,
        is_anchor=is_anchor,
    )


def enforce_monotonicity(blocks, phi, weights, wg, r, Ytilde, xi, I):
    """
    Standard adjacent-block merging step.
    """
    stack: list[Block] = []

    for block in blocks:
        stack.append(block)

        while len(stack) >= 2 and stack[-2].wage > stack[-1].wage + PAV_TOL:
            right_block = stack.pop()
            left_block = stack.pop()

            merged = merge_two_blocks(
                left_block,
                right_block,
                phi,
                weights,
                wg,
                r,
                Ytilde,
                xi,
                I,
            )

            stack.append(merged)

    return stack


def expert_part_of_block(block, n_anchor):
    """
    Return the cell interval of the expert part of a block.

    The generic part [0,I] is not subject to expertise-ladder feasibility.
    """
    left = max(block.left, n_anchor)
    right = block.right

    if right <= left:
        return None

    return left, right


def block_expert_feasibility(block, obj_like):
    """
    Cumulative feasibility inside the expert part of one block:

        int_z^b L(phi)dphi <= (b-z)Lbar

    Returns min slack and the split index that would repair the worst violation.
    """
    n_anchor = obj_like["n_anchor"]
    edges = obj_like["edges"]
    weights = obj_like["weights"]
    L = obj_like["L"]

    expert_part = expert_part_of_block(block, n_anchor)

    if expert_part is None:
        return dict(
            feasible=True,
            min_slack=0.0,
            split_idx=None,
            argmin_phi=None,
        )

    left, right = expert_part

    demand_mass = L[left:right] * weights[left:right]

    demand_right = np.zeros(right - left + 1)
    demand_right[:-1] = np.cumsum(demand_mass[::-1])[::-1]
    demand_right[-1] = 0.0

    local_edges = edges[left:right + 1]
    block_right_edge = edges[right]

    supply_right = (block_right_edge - local_edges) * Lbar
    slack = supply_right - demand_right

    idx = int(np.argmin(slack))
    min_slack = float(slack[idx])
    split_idx = left + idx

    # A split at the right edge is meaningless.
    if split_idx >= right:
        split_idx = None

    return dict(
        feasible=min_slack >= -FEAS_TOL,
        min_slack=min_slack,
        split_idx=split_idx,
        argmin_phi=float(local_edges[idx]),
    )


def split_block_for_feasibility(block, split_idx, phi, weights, wg, r, Ytilde, xi, I):
    """
    Split a block at split_idx.

    If the original block is anchor, the left part remains anchor at w_g.
    The right part becomes a non-anchor expert block.

    If the original block is non-anchor, both parts solve their own block wages.
    """
    if split_idx is None or split_idx <= block.left or split_idx >= block.right:
        return [block]

    if block.is_anchor:
        left_block = Block(
            left=block.left,
            right=split_idx,
            wage=wg,
            is_anchor=True,
        )

        right_wage = solve_block_wage(
            phi,
            weights,
            split_idx,
            block.right,
            r,
            Ytilde,
            xi,
            I,
        )

        right_block = Block(
            left=split_idx,
            right=block.right,
            wage=right_wage,
            is_anchor=False,
        )

        return [left_block, right_block]

    left_wage = solve_block_wage(
        phi,
        weights,
        block.left,
        split_idx,
        r,
        Ytilde,
        xi,
        I,
    )

    right_wage = solve_block_wage(
        phi,
        weights,
        split_idx,
        block.right,
        r,
        Ytilde,
        xi,
        I,
    )

    return [
        Block(block.left, split_idx, left_wage, is_anchor=False),
        Block(split_idx, block.right, right_wage, is_anchor=False),
    ]


def refine_blocks_for_expert_feasibility(blocks, phi, weights, edges, n_anchor, wg, r, Ytilde, xi, I):
    """
    Iteratively split blocks that violate expert cumulative feasibility,
    then re-enforce monotonicity.

    This is the key correction relative to ordinary PAV.
    """
    n = phi.size

    for _ in range(MAX_FEAS_REFINEMENTS):
        wages = assign_wages_from_blocks(n, blocks)
        L = labor_demand(phi, wages, r, Ytilde, xi, I)

        obj_like = dict(
            L=L,
            weights=weights,
            edges=edges,
            n_anchor=n_anchor,
        )

        worst_idx = None
        worst_diag = None

        for k, block in enumerate(blocks):
            diag = block_expert_feasibility(block, obj_like)

            if worst_diag is None or diag["min_slack"] < worst_diag["min_slack"]:
                worst_diag = diag
                worst_idx = k

        if worst_diag is None or worst_diag["min_slack"] >= -FEAS_TOL:
            return blocks

        block = blocks[worst_idx]
        split_idx = worst_diag["split_idx"]

        if split_idx is None:
            return blocks

        new_pieces = split_block_for_feasibility(
            block,
            split_idx,
            phi,
            weights,
            wg,
            r,
            Ytilde,
            xi,
            I,
        )

        blocks = blocks[:worst_idx] + new_pieces + blocks[worst_idx + 1:]
        blocks = enforce_monotonicity(blocks, phi, weights, wg, r, Ytilde, xi, I)

    raise RuntimeError(
        f"Feasibility refinement failed after {MAX_FEAS_REFINEMENTS} iterations."
    )


def anchored_feasible_pav_wage_schedule(phi, weights, edges, n_anchor, wg, r, Ytilde, xi, I):
    """
    Anchored PAV with expert cumulative feasibility.

    Initial structure:
        anchor block: [0,I], wage fixed at w_g
        expert cells: singleton blocks with unconstrained expert wages

    Steps:
        1. enforce monotonicity by adjacent pooling;
        2. split blocks that violate expert downward feasibility;
        3. re-enforce monotonicity after each split.
    """
    n = phi.size

    if n_anchor <= 0:
        raise RuntimeError("Anchor block is empty. Check I and grid construction.")

    if n_anchor >= n:
        raise RuntimeError("No expert region. Need I < 1.")

    w_pre = np.full(n, np.nan, dtype=float)
    w_pre[:n_anchor] = wg

    w_exp = unconstrained_expert_wage(phi[n_anchor:], r, Ytilde, xi, I)

    if np.any(~np.isfinite(w_exp)) or np.any(w_exp <= 0.0):
        raise RuntimeError("Invalid expert unconstrained wage schedule.")

    w_pre[n_anchor:] = w_exp

    blocks: list[Block] = [
        Block(left=0, right=n_anchor, wage=wg, is_anchor=True)
    ]

    for i in range(n_anchor, n):
        blocks.append(Block(i, i + 1, float(w_pre[i]), is_anchor=False))

    blocks = enforce_monotonicity(blocks, phi, weights, wg, r, Ytilde, xi, I)

    blocks = refine_blocks_for_expert_feasibility(
        blocks,
        phi,
        weights,
        edges,
        n_anchor,
        wg,
        r,
        Ytilde,
        xi,
        I,
    )

    wages = assign_wages_from_blocks(n, blocks)

    return wages, blocks, w_pre


# ------------------------------------------------------------------
# Equilibrium objects and residuals
# ------------------------------------------------------------------

def equilibrium_objects(log_x, I, n_grid=DEFAULT_N_GRID):
    """
    log_x = (log w_g, log r, log Ytilde, log xi)
    """
    wg, r, Ytilde, xi = np.exp(log_x)

    phi, weights, edges, n_anchor = make_task_grid(I, n_grid=n_grid)

    wages, blocks, w_pre = anchored_feasible_pav_wage_schedule(
        phi,
        weights,
        edges,
        n_anchor,
        wg,
        r,
        Ytilde,
        xi,
        I,
    )

    a = alpha(phi, I)
    om1_lam, om2_lam, OM = omega_components(phi, xi)

    p = task_price(wages, r, a)
    p_power = p ** (1.0 - lam)

    L = labor_demand(phi, wages, r, Ytilde, xi, I)
    K = capital_demand(phi, wages, r, Ytilde, xi, I)

    L1 = om1_lam / OM * L
    L2 = xi * om2_lam / OM * L

    Q1_integrand = om1_lam * p_power
    Q2_integrand = om2_lam * p_power

    L_total = float(np.dot(weights, L))
    K_total = float(np.dot(weights, K))
    Q1 = float(np.dot(weights, Q1_integrand))
    Q2 = float(np.dot(weights, Q2_integrand))

    anchor_block = blocks[0]

    if not anchor_block.is_anchor:
        raise RuntimeError("First block is not anchor. PAV logic failed.")

    L_anchor = float(
        np.dot(
            weights[anchor_block.left:anchor_block.right],
            L[anchor_block.left:anchor_block.right],
        )
    )

    supply_anchor = block_supply(weights, anchor_block.left, anchor_block.right)

    return dict(
        I=I,
        wg=wg,
        r=r,
        Ytilde=Ytilde,
        xi=xi,
        phi=phi,
        weights=weights,
        edges=edges,
        n_anchor=n_anchor,
        wages=wages,
        w_pre=w_pre,
        blocks=blocks,
        L=L,
        L1=L1,
        L2=L2,
        K=K,
        Q1=Q1,
        Q2=Q2,
        L_total=L_total,
        K_total=K_total,
        anchor_left=anchor_block.left,
        anchor_right=anchor_block.right,
        anchor_right_phi=edges[anchor_block.right],
        L_anchor=L_anchor,
        supply_anchor=supply_anchor,
    )


def residuals(log_x, I, n_grid=DEFAULT_N_GRID):
    obj = equilibrium_objects(log_x, I, n_grid=n_grid)

    res_anchor_labor = (
        obj["supply_anchor"] - obj["L_anchor"]
    ) / max(abs(obj["supply_anchor"]), EPS)

    res_capital = (Kbar - obj["K_total"]) / max(abs(Kbar), EPS)

    res_price1 = 1.0 - obj["Q1"]

    res_price2 = (
        obj["xi"] - 1.0 / obj["Q2"]
    ) / max(abs(obj["xi"]), EPS)

    return np.array(
        [
            res_anchor_labor,
            res_capital,
            res_price1,
            res_price2,
        ],
        dtype=float,
    )


# ------------------------------------------------------------------
# Solver
# ------------------------------------------------------------------

def solve_equilibrium(I, n_grid=DEFAULT_N_GRID, verbose=False):
    seeds = [
        (0.0, -3.0, 0.5, 0.0),
        (0.0, -3.0, 0.0, 0.0),
        (0.2, -3.5, 0.5, 0.5),
        (-0.2, -2.5, 0.0, -0.5),
        (0.5, -2.0, 0.5, 0.5),
        (0.0, -3.0, 0.7, -0.2),
        (0.0, -2.5, 0.3, 0.0),
    ]

    best = None
    failures = []

    for seed in seeds:
        x0 = np.array(seed, dtype=float)

        try:
            sol = root(
                lambda x: residuals(x, I, n_grid=n_grid),
                x0,
                method="hybr",
                tol=ROOT_TOL,
            )

            err = float(np.linalg.norm(sol.fun))

            if verbose:
                print(
                    f"[outer] I={I:.4f}, seed={seed}, "
                    f"success={sol.success}, err={err:.3e}, message={sol.message}"
                )

            candidate = dict(
                x=sol.x,
                err=err,
                success=bool(sol.success),
                message=sol.message,
            )

            if best is None or candidate["err"] < best["err"]:
                best = candidate

            if sol.success and err < ROOT_TOL:
                break

        except Exception as e:
            failures.append((seed, repr(e)))

            if verbose:
                print(f"[outer] Seed failed for I={I:.4f}, seed={seed}")
                print(f"        Error: {repr(e)}")

    if best is None:
        msg = f"All outer solver seeds failed for I={I}."

        if failures:
            msg += "\nSeed failures:\n" + "\n".join(
                f"  {seed}: {err}" for seed, err in failures
            )

        raise RuntimeError(msg)

    if best["err"] > 1e-6:
        msg = (
            f"Outer solver did not converge tightly for I={I}. "
            f"Best err={best['err']:.3e}, message={best['message']}"
        )

        if failures:
            msg += "\nSeed failures:\n" + "\n".join(
                f"  {seed}: {err}" for seed, err in failures
            )

        raise RuntimeError(msg)

    obj = equilibrium_objects(best["x"], I, n_grid=n_grid)

    obj["x"] = best["x"]
    obj["err"] = best["err"]
    obj["residuals"] = residuals(best["x"], I, n_grid=n_grid)

    P2 = obj["xi"] ** (1.0 / (lam - 1.0))

    obj["P2"] = P2
    obj["Y1"] = obj["Ytilde"]
    obj["Y2"] = obj["Ytilde"] / P2

    return obj


# ------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------

def block_table(obj):
    rows = []
    edges = obj["edges"]

    for k, block in enumerate(obj["blocks"]):
        rows.append(
            dict(
                block=k,
                left=float(edges[block.left]),
                right=float(edges[block.right]),
                wage=float(block.wage),
                length=float(edges[block.right] - edges[block.left]),
                n_cells=block.size,
                is_anchor=block.is_anchor,
            )
        )

    return rows


def employment_table(obj, threshold):
    """
    Compute sector employment split by inexpert (phi<=threshold) and expert (phi>threshold).
    Returns dict with keys L1, L2, L; each has total/inexpert/expert.
    """
    phi = obj["phi"]
    weights = obj["weights"]
    wages = obj["wages"]
    r = obj["r"]
    Ytilde = obj["Ytilde"]
    xi = obj["xi"]
    n_anchor = obj["n_anchor"]

    a = alpha(phi, threshold)
    om1_lam, om2_lam, OM = omega_components(phi, xi)

    p = task_price(wages, r, a)
    L = (1.0 - a) / wages * OM * p ** (1.0 - lam) * Ytilde

    L1 = om1_lam / OM * L
    L2 = xi * om2_lam / OM * L

    def intg(v, sl):
        return float(np.dot(weights[sl], v[sl]))

    inexp = slice(None, n_anchor)
    exp = slice(n_anchor, None)

    rows = {}
    for key, L_s in [("L1", L1), ("L2", L2)]:
        rows[key] = dict(
            total=intg(L_s, slice(None)),
            inexpert=intg(L_s, inexp),
            expert=intg(L_s, exp),
        )

    rows["L"] = dict(
        total=rows["L1"]["total"] + rows["L2"]["total"],
        inexpert=rows["L1"]["inexpert"] + rows["L2"]["inexpert"],
        expert=rows["L1"]["expert"] + rows["L2"]["expert"],
    )

    return rows


def check_expert_downward_feasibility(obj):
    """
    Check expert-only cumulative feasibility:

        int_z^1 L(phi)dphi <= (1-z)Lbar

    only for z >= I.
    """
    L = obj["L"]
    weights = obj["weights"]
    edges = obj["edges"]
    n_anchor = obj["n_anchor"]

    demand_mass = L * weights

    demand_right = np.zeros(edges.size)
    demand_right[:-1] = np.cumsum(demand_mass[::-1])[::-1]
    demand_right[-1] = 0.0

    supply_right = (1.0 - edges) * Lbar
    slack = supply_right - demand_right

    expert_edges = edges[n_anchor:]
    expert_slack = slack[n_anchor:]

    idx_local = int(np.argmin(expert_slack))
    idx = n_anchor + idx_local

    return dict(
        min_slack=float(slack[idx]),
        argmin_phi=float(edges[idx]),
        slack=slack,
        expert_edges=expert_edges,
        expert_slack=expert_slack,
        demand_right=demand_right,
        supply_right=supply_right,
    )


def check_block_downward_feasibility(obj):
    edges = obj["edges"]
    L = obj["L"]
    weights = obj["weights"]
    n_anchor = obj["n_anchor"]

    diagnostics = []

    for k, block in enumerate(obj["blocks"]):
        expert_part = expert_part_of_block(block, n_anchor)

        if expert_part is None:
            diagnostics.append(
                dict(
                    block=k,
                    left=float(edges[block.left]),
                    right=float(edges[block.right]),
                    wage=float(block.wage),
                    is_anchor=block.is_anchor,
                    min_slack=0.0,
                    argmin_phi=None,
                )
            )
            continue

        left, right = expert_part
        demand_mass = L[left:right] * weights[left:right]

        demand_right = np.zeros(right - left + 1)
        demand_right[:-1] = np.cumsum(demand_mass[::-1])[::-1]
        demand_right[-1] = 0.0

        local_edges = edges[left:right + 1]
        block_right_edge = edges[right]

        supply_right = (block_right_edge - local_edges) * Lbar
        slack = supply_right - demand_right

        idx = int(np.argmin(slack))

        diagnostics.append(
            dict(
                block=k,
                left=float(edges[block.left]),
                right=float(edges[block.right]),
                wage=float(block.wage),
                is_anchor=block.is_anchor,
                min_slack=float(slack[idx]),
                argmin_phi=float(local_edges[idx]),
            )
        )

    return diagnostics


def check_monotonicity(obj):
    w = obj["wages"]
    diffs = np.diff(w)

    min_diff = float(np.min(diffs)) if diffs.size else 0.0

    return dict(
        min_diff=min_diff,
        is_nondecreasing=bool(min_diff >= -1e-9),
    )


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

I_VALUES = [0.25, 0.50]
I_COLORS = {0.25: "#238b45", 0.50: "#6a51a3"}
I_STYLES = {0.25: "-", 0.50: "--"}
WEIGHT_TEXT = (
    rf"$\omega_1(\phi)=1-{LINEAR_SLOPE:.2f}+{2.0 * LINEAR_SLOPE:.2f}\phi$, "
    rf"$\omega_2(\phi)=1+{LINEAR_SLOPE:.2f}-{2.0 * LINEAR_SLOPE:.2f}\phi$"
)


def mark_frontier(ax, I, color, label=None):
    frontier_color = "#d95f5f"
    ax.axvline(
        I,
        color=frontier_color,
        ls="--",
        lw=1.0,
        alpha=0.65,
        label=label,
        zorder=2,
    )


def plot_wages(results, fname=None, show=False):
    fig, ax = plt.subplots(figsize=(8.8, 5.6))

    for I in I_VALUES:
        obj = results[I]
        color = I_COLORS[I]

        ax.plot(
            obj["phi"],
            obj["wages"],
            color=color,
            ls=I_STYLES[I],
            lw=2.4,
            label=(
                fr"$I={I:.2f}$ feasible anchored PAV "
                fr"($w_g={obj['wg']:.3f}$, blocks={len(obj['blocks'])})"
            ),
        )

        ax.plot(
            obj["phi"],
            obj["w_pre"],
            color=color,
            ls=":",
            lw=1.3,
            alpha=0.9,
            label=fr"$I={I:.2f}$ pre-PAV wage",
        )

        mark_frontier(ax, I, color)
        ax.axhline(obj["wg"], color=color, ls="-.", lw=0.7, alpha=0.45)

    ax.set_xlabel(r"occupation $\phi$")
    ax.set_ylabel(r"task wage $w(\phi)$")
    ax.set_title("Feasible anchored downward-mobility PAV occupation wages [Aggregate]")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, frameon=False, loc="best")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)

    if fname is not None:
        fig.savefig(fname, dpi=150)

    if not show:
        plt.close(fig)


def plot_sector_wages(results, fname=None, show=False):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), sharey=True)
    sector_titles = ["Sector 1", "Sector 2"]

    for ax, sector_title in zip(axes, sector_titles):
        for I in I_VALUES:
            obj = results[I]
            color = I_COLORS[I]

            ax.plot(
                obj["phi"],
                obj["wages"],
                color=color,
                ls=I_STYLES[I],
                lw=2.4,
                label=fr"$I={I:.2f}$ PAV wage",
            )

            ax.axhline(
                obj["wg"],
                color=color,
                ls="-.",
                lw=0.8,
                alpha=0.55,
                label=fr"$I={I:.2f}$ $w_g={obj['wg']:.3f}$",
            )

            mark_frontier(ax, I, color)

        ax.set_title(sector_title)
        ax.set_xlabel(r"occupation $\phi$")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, frameon=False, loc="best")

    axes[0].set_ylabel(r"task wage $w(\phi)$")
    fig.suptitle("Sector occupation wages under feasible anchored downward-mobility PAV", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)

    if fname is not None:
        fig.savefig(fname, dpi=150)

    if not show:
        plt.close(fig)


def plot_total_labor(results, fname=None, show=False):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), sharey=True)
    sector_specs = [
        (
            rf"Sector 1: $\omega_1(\phi)=1-{LINEAR_SLOPE:.2f}+{2.0 * LINEAR_SLOPE:.2f}\phi$",
            "L1",
            r"$\ell_1(\phi)$",
        ),
        (
            rf"Sector 2: $\omega_2(\phi)=1+{LINEAR_SLOPE:.2f}-{2.0 * LINEAR_SLOPE:.2f}\phi$",
            "L2",
            r"$\ell_2(\phi)$",
        ),
    ]

    for ax, (sector_title, labor_key, ylabel) in zip(axes, sector_specs):
        for I in I_VALUES:
            obj = results[I]
            color = I_COLORS[I]

            ax.plot(
                obj["phi"],
                obj[labor_key],
                color=color,
                ls=I_STYLES[I],
                lw=2.2,
                label=fr"$I={I:.2f}$ {ylabel}",
            )

            mark_frontier(ax, I, color)

        ax.axhline(Lbar, color="k", lw=0.7, alpha=0.35, label=r"$\bar L$")
        ax.set_title(sector_title)
        ax.set_xlabel(r"occupation $\phi$")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, frameon=False, loc="best")

    axes[0].set_ylabel(r"$L(\phi)$")
    fig.suptitle(r"Equilibrium sector-occupation employment densities by frontier $I$", fontsize=12)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)

    if fname is not None:
        fig.savefig(fname, dpi=150)

    if not show:
        plt.close(fig)


def plot_labor(results, fname=None, show=False):
    fig, axes = plt.subplots(1, len(I_VALUES), figsize=(14, 5.0), sharey=True)

    if len(I_VALUES) == 1:
        axes = [axes]

    for ax, I in zip(axes, I_VALUES):
        obj = results[I]

        ax.plot(
            obj["phi"],
            obj["L"],
            color="k",
            lw=2.0,
            ls="--",
            label=r"total labor $L(\phi)$",
        )

        ax.plot(
            obj["phi"],
            obj["L1"],
            color="#08306b",
            lw=1.8,
            label=r"sector 1 $\ell_1(\phi)$",
        )

        ax.plot(
            obj["phi"],
            obj["L2"],
            color="#a50f15",
            lw=1.8,
            ls="-",
            label=r"sector 2 $\ell_2(\phi)$",
        )

        ax.axhline(Lbar, color="k", lw=0.6, alpha=0.35, label=r"$\bar L$")
        mark_frontier(ax, I, I_COLORS[I])

        ax.axvline(
            obj["anchor_right_phi"],
            color="k",
            ls="--",
            lw=0.8,
            alpha=0.5,
        )

        shaded_right = None

        for block in obj["blocks"]:
            if block.size > 1:
                left = obj["edges"][block.left]
                right = obj["edges"][block.right]
                ax.axvspan(left, right, alpha=0.08)
                shaded_right = right

        if shaded_right is not None:
            ax.annotate(
                fr"$\phi={shaded_right:.3f}$",
                xy=(shaded_right, 0.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, 8),
                textcoords="offset points",
                va="bottom",
                ha="center",
                fontsize=8.5,
                color="0.25",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="0.65", alpha=0.78),
            )

        ax.set_title(
            fr"$I={I:.2f}$, $w_g={obj['wg']:.3f}$"
        )
        ax.set_xlabel(r"occupation $\phi$")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, frameon=False, loc="best")

    axes[0].set_ylabel(r"$L(\phi)$")
    fig.suptitle("Employment under feasible anchored downward-mobility PAV [Aggregate]", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)

    if fname is not None:
        fig.savefig(fname, dpi=150)

    if not show:
        plt.close(fig)


def plot_expert_feasibility(results, fname=None, show=False):
    fig, ax = plt.subplots(figsize=(8.8, 5.6))

    for I in I_VALUES:
        obj = results[I]
        diag = check_expert_downward_feasibility(obj)

        ax.plot(
            diag["expert_edges"],
            diag["expert_slack"],
            color=I_COLORS[I],
            ls=I_STYLES[I],
            lw=2.0,
            label=fr"$I={I:.2f}$ min expert slack={diag['min_slack']:.2e}",
        )

        mark_frontier(ax, I, I_COLORS[I])

    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"expert cutoff $z \geq I$")
    ax.set_ylabel(r"$(1-z)\bar L - \int_z^1 L(\phi)d\phi$")
    ax.set_title("Expert downward-mobility cumulative feasibility [Aggregate]")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, frameon=False, loc="best")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)

    if fname is not None:
        fig.savefig(fname, dpi=150)

    if not show:
        plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.perf_counter()
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    print("\n=== Feasible anchored downward-mobility PAV equilibrium ===")

    hdr = (
        f"{'I':>5}"
        f"{'w_g':>10}"
        f"{'r':>10}"
        f"{'Ytilde':>10}"
        f"{'P_2':>10}"
        f"{'xi':>10}"
        f"{'Y_1':>10}"
        f"{'Y_2':>10}"
        f"{'blocks':>8}"
        f"{'anchor_R':>10}"
        f"{'L_total':>11}"
        f"{'K_total':>11}"
        f"{'err':>12}"
    )

    print(hdr)
    print("-" * len(hdr))

    for I in I_VALUES:
        obj = solve_equilibrium(I, n_grid=DEFAULT_N_GRID, verbose=False)
        results[I] = obj

        print(
            f"{I:>5.2f}"
            f"{obj['wg']:>10.4f}"
            f"{obj['r']:>10.4f}"
            f"{obj['Ytilde']:>10.4f}"
            f"{obj['P2']:>10.4f}"
            f"{obj['xi']:>10.4f}"
            f"{obj['Y1']:>10.4f}"
            f"{obj['Y2']:>10.4f}"
            f"{len(obj['blocks']):>8d}"
            f"{obj['anchor_right_phi']:>10.4f}"
            f"{obj['L_total']:>11.6f}"
            f"{obj['K_total']:>11.6f}"
            f"{obj['err']:>12.2e}"
        )

        print(f"      residuals = {np.array2string(obj['residuals'], precision=3)}")

        print(
            f"      anchor block: [0.0000, {obj['anchor_right_phi']:.4f}], "
            f"L_anchor={obj['L_anchor']:.6f}, "
            f"supply_anchor={obj['supply_anchor']:.6f}"
        )

        mono = check_monotonicity(obj)
        print(
            f"      monotonicity: nondecreasing={mono['is_nondecreasing']}, "
            f"min diff={mono['min_diff']:.2e}"
        )

        expert_feas = check_expert_downward_feasibility(obj)
        print(
            f"      expert feasibility: min slack={expert_feas['min_slack']:.2e} "
            f"at z={expert_feas['argmin_phi']:.4f}"
        )

        block_feas = check_block_downward_feasibility(obj)
        min_block = min(block_feas, key=lambda d: d["min_slack"])

        print(
            f"      worst block feasibility: block={min_block['block']}, "
            f"[{min_block['left']:.4f}, {min_block['right']:.4f}], "
            f"anchor={min_block['is_anchor']}, "
            f"min slack={min_block['min_slack']:.2e}"
        )

        print("      first blocks:")

        for row in block_table(obj)[:8]:
            print(
                f"        #{row['block']:>3d}: "
                f"[{row['left']:.4f}, {row['right']:.4f}], "
                f"w={row['wage']:.4f}, "
                f"cells={row['n_cells']}, "
                f"anchor={row['is_anchor']}"
            )

        if len(obj["blocks"]) > 8:
            print(f"        ... {len(obj['blocks']) - 8} more blocks")

    print("\nP2 and xi by I:")
    px_hdr = f"{'I':>5}{'P_2':>12}{'xi':>12}"
    print(px_hdr)
    print("-" * len(px_hdr))

    for I in I_VALUES:
        obj = results[I]
        print(f"{I:>5.2f}{obj['P2']:>12.4f}{obj['xi']:>12.4f}")

    print("\nEmployment by sector:")
    emp_hdr = (
        f"{'I':>5}{'sector':>10}{'total emp':>12}"
        f"{'inexpert emp':>15}{'expert emp':>13}"
    )
    print(emp_hdr)
    print("-" * len(emp_hdr))

    sector_names = {"L1": "Sector 1", "L2": "Sector 2", "L": "Total"}

    for I in I_VALUES:
        emp = employment_table(results[I], I)
        for key in ("L1", "L2", "L"):
            row = emp[key]
            print(
                f"{I:>5.2f}{sector_names[key]:>10}{row['total']:>12.4f}"
                f"{row['inexpert']:>15.4f}{row['expert']:>13.4f}"
            )

    plot_wages(
        results,
        GRAPH_DIR / "7. TwoSector_sameI_wages.png",
        show=True,
    )

    plot_total_labor(
        results,
        GRAPH_DIR / "7. TwoSector_sameI_Lphi.png",
        show=True,
    )

    plot_labor(
        results,
        GRAPH_DIR / "7. TwoSector_sameI_labor.png",
        show=True,
    )

    plot_expert_feasibility(
        results,
        GRAPH_DIR / "7. TwoSector_sameI_feasibility.png",
        show=False,
    )

    print(f"\nTotal time: {time.perf_counter() - t0:.2f}s")
    print("Wrote:")
    print(f"  {GRAPH_DIR / '7. TwoSector_sameI_wages.png'}")
    print(f"  {GRAPH_DIR / '7. TwoSector_sameI_Lphi.png'}")
    print(f"  {GRAPH_DIR / '7. TwoSector_sameI_labor.png'}")
    print(f"  {GRAPH_DIR / '7. TwoSector_sameI_feasibility.png'}")
    plt.show()
