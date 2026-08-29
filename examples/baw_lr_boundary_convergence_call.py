from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from src.pricers import BAW_quadratic, BAW_call_critical_spot
from src.pricers import OptionSpec
import numpy as np
import matplotlib.pyplot as plt
from linker_cpp import discrete_models
from scipy.optimize import brentq

q_grid = np.array([0.05, 0.10,0.20, 0.35, 0.45, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00])
steps_grid = np.array([1001, 3001, 5001, 10001])

@dataclass
class BoundaryConvergenceResult:
    q: float
    baw_boundaries: float
    lr_boundaries: np.ndarray
    fit_slope: float
    lr_boundary_inf: float
    r_squared: float

def lr_gap_at_spot(spot:float, spec:OptionSpec,
                   option: discrete_models.VanillaOption, config: discrete_models.DiscreteModel):
    """
    Root exercise gap for the LR American tree:

        gap(S) = continuation_value(S) - intrinsic_value(S)

    For an American call:

        gap > 0  -> continuation
        gap = 0  -> exercise boundary
        gap < 0  -> immediate exercise
    """
    bumped_market = discrete_models.MarketData(
        float(spot),
        spec.rate,
        spec.dividend_yield,
        spec.volatility,
    )

    return discrete_models.LR_root(
        option,
        bumped_market,
        config,
    )

def lr_call_critical_spot(
    spec: OptionSpec,
    option: discrete_models.VanillaOption,
    config: discrete_models.TreeConfig,
    initial_guess: float,
) -> float:
    """
    Find S*_LR such that:

        continuation_value(S*) - intrinsic_value(S*) = 0.

    The initial guess is normally the BAW boundary for the first N,
    then the previous LR boundary when N increases.
    """

    relative_width = 0.02

    lower = max(
        spec.strike,
        initial_guess * (1.0 - relative_width),
    )

    upper = initial_guess * (1.0 + relative_width)

    f_lower = lr_gap_at_spot(
        lower,
        spec,
        option,
        config,
    )

    f_upper = lr_gap_at_spot(
        upper,
        spec,
        option,
        config,
    )

    expansion_count = 0
    max_expansions = 20

    while f_lower * f_upper > 0.0:

        relative_width *= 1.5

        lower = max(
            spec.strike,
            initial_guess * (1.0 - relative_width),
        )

        upper = initial_guess * (1.0 + relative_width)

        f_lower = lr_gap_at_spot(
            lower,
            spec,
            option,
            config,
        )

        f_upper = lr_gap_at_spot(
            upper,
            spec,
            option,
            config,
        )

        expansion_count += 1

        if expansion_count >= max_expansions:
            raise RuntimeError(
                "Unable to bracket LR exercise boundary "
                f"for q={spec.dividend_yield:.6f}, "
                f"N={config.steps}"
            )
    return float(
        brentq(
            lambda s: lr_gap_at_spot(
                s,
                spec,
                option,
                config,
            ),
            lower,
            upper,
            xtol=1e-4,
            rtol=1e-8,
            maxiter=50,
        )
    )

def extrapolate_boundary(
    steps_grid: np.ndarray,
    boundaries: np.ndarray,
) -> tuple[float, float, float]:
    """
    Empirical fit:

        S*_LR(N) = intercept + slope / sqrt(N)

    Therefore:

        intercept = S*_LR(N -> infinity)

    Returns
    -------
    slope
    intercept
    R²
    """

    x = 1.0 / np.sqrt(
        steps_grid.astype(float)
    )

    y = boundaries.astype(float)

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    fitted = slope * x + intercept

    ss_res = np.sum(
        (y - fitted) ** 2
    )

    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    if ss_tot > 0.0:
        r_squared = 1.0 - ss_res / ss_tot
    else:
        r_squared = 1.0

    return (
        float(slope),
        float(intercept),
        float(r_squared),
    )

def run_one_q(
    q: float,
) -> BoundaryConvergenceResult:

    spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        steps=1001,
        option_type="Call",
        exercise_style="American",
        model="Leisen-Reimer",
        dividend_yield=float(q),
    )

    payoff = discrete_models.PayoffCall(
        spec.strike
    )

    option = discrete_models.VanillaOption(
        payoff,
        spec.maturity,
        discrete_models.ExerciseStyle.American,
    )

    Q2 = BAW_quadratic(spec)[1]

    baw_boundary = BAW_call_critical_spot(
        spec,
        Q2,
    )

    lr_boundaries = []

    # First seed = BAW boundary.
    # Afterward we use the preceding LR boundary:
    # much tighter bracket, hence fewer expensive tree evaluations.
    boundary_guess = baw_boundary

    print()
    print("=" * 78)
    print(
        f"q = {q:.3f} | "
        f"S*_BAW = {baw_boundary:.6f}"
    )
    print("=" * 78)

    for steps in steps_grid:

        config = discrete_models.TreeConfig(
            int(steps)
        )

        start = perf_counter()

        s_star_lr = lr_call_critical_spot(
            spec,
            option,
            config,
            initial_guess=boundary_guess,
        )

        elapsed = perf_counter() - start

        lr_boundaries.append(
            s_star_lr
        )

        print(
            f"N={steps:5d} | "
            f"S*_LR={s_star_lr:.6f} | "
            f"BAW-LR={baw_boundary - s_star_lr:+.6f} | "
            f"time={elapsed:.3f}s"
        )

        # Excellent seed for next N
        boundary_guess = s_star_lr

    lr_boundaries_array = np.asarray(
        lr_boundaries,
        dtype=float,
    )

    slope, boundary_inf, r_squared = (
        extrapolate_boundary(
            steps_grid,
            lr_boundaries_array,
        )
    )

    print("-" * 78)

    print(
        f"LR extrapolated N->inf : "
        f"{boundary_inf:.6f}"
    )

    print(
        f"BAW boundary           : "
        f"{baw_boundary:.6f}"
    )

    print(
        f"BAW - LR(inf)          : "
        f"{baw_boundary - boundary_inf:+.6f}"
    )

    print(
        f"R²                     : "
        f"{r_squared:.10f}"
    )

    return BoundaryConvergenceResult(
        q=float(q),
        baw_boundaries=float(baw_boundary),
        lr_boundaries=lr_boundaries_array,
        fit_slope=slope,
        lr_boundary_inf=boundary_inf,
        r_squared=r_squared,
    )

def run_boundary_convergence_study(
) -> list[BoundaryConvergenceResult]:

    results = []

    study_start = perf_counter()

    for q in q_grid:

        result = run_one_q(
            float(q)
        )

        results.append(
            result
        )

    total_time = (
        perf_counter()
        - study_start
    )

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    for result in results:

        print(
            f"q={result.q:.3f} | "
            f"S*_BAW={result.baw_boundaries:.6f} | "
            f"S*_LR(inf)={result.lr_boundary_inf:.6f} | "
            f"dS*={result.baw_boundaries - result.lr_boundary_inf:+.6f} | "
            f"R²={result.r_squared:.8f}"
        )

    print()
    print(
        f"Total study time: "
        f"{total_time:.3f}s"
    )

    return results

def plot_boundary_convergence(
    results: list[BoundaryConvergenceResult],
) -> None:
    """
    For each q:

        x = 1 / sqrt(N)
        y = S*_LR(N)

    The fitted straight line is extrapolated to x = 0,
    corresponding to N -> infinity.

    The BAW boundary is shown as a horizontal reference.
    """

    x = 1.0 / np.sqrt(
        steps_grid.astype(float)
    )

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(15, 9),
        constrained_layout=True,
    )

    axes = axes.flatten()

    for ax, result in zip(
        axes,
        results,
    ):

        # -------------------------------------------------------------------
        # LR discrete boundaries
        # -------------------------------------------------------------------

        ax.scatter(
            x,
            result.lr_boundaries,
            label="LR",
            zorder=3,
        )

        # -------------------------------------------------------------------
        # Linear fit, including x = 0
        # -------------------------------------------------------------------

        x_fit = np.linspace(
            0.0,
            x.max() * 1.05,
            200,
        )

        y_fit = (
            result.fit_slope * x_fit
            + result.lr_boundary_inf
        )

        ax.plot(
            x_fit,
            y_fit,
            label="Linear fit",
        )

        # -------------------------------------------------------------------
        # N -> infinity point
        # -------------------------------------------------------------------

        ax.scatter(
            [0.0],
            [result.lr_boundary_inf],
            marker="x",
            s=80,
            label="LR extrapolated",
            zorder=4,
        )

        # -------------------------------------------------------------------
        # BAW boundary
        # -------------------------------------------------------------------

        ax.axhline(
            result.baw_boundaries,
            linestyle="--",
            label="BAW",
        )

        # -------------------------------------------------------------------
        # Annotate N
        # -------------------------------------------------------------------

        for x_i, y_i, steps in zip(
            x,
            result.lr_boundaries,
            steps_grid,
        ):

            ax.annotate(
                f"{steps}",
                (x_i, y_i),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

        # -------------------------------------------------------------------
        # Formatting
        # -------------------------------------------------------------------

        boundary_error = (
            result.baw_boundaries
            - result.lr_boundary_inf
        )

        ax.set_title(
            f"q = {result.q:.2f}\n"
            f"BAW - LR(inf) = {boundary_error:+.4f}"
        )

        ax.set_xlabel(
            "1 / sqrt(N)"
        )

        ax.set_ylabel(
            "Critical spot S*"
        )

        ax.grid(
            True
        )

        ax.legend(
            fontsize=8
        )

        # R² inside graph
        ax.text(
            0.03,
            0.05,
            f"R² = {result.r_squared:.6f}",
            transform=ax.transAxes,
            fontsize=9,
        )

    fig.suptitle(
        "Leisen-Reimer American Call Exercise Boundary\n"
        "Linear Extrapolation in 1 / sqrt(N)",
        fontsize=14,
    )

    plt.show()

def plot_boundary_error(
    results: list[BoundaryConvergenceResult],
) -> None:

    q_values = np.array([
        result.q
        for result in results
    ])

    boundary_errors = np.array([
        result.baw_boundaries - result.lr_boundary_inf
        for result in results
    ])

    plt.figure(figsize=(10, 6))

    plt.plot(
        q_values,
        boundary_errors,
        marker="o",
        label="BAW - LR extrapolated",
    )

    plt.axhline(
        0.0,
        linestyle="--",
    )

    plt.xlabel("Dividend yield q")
    plt.ylabel("Boundary error ΔS*")
    plt.title(
        "BAW Exercise Boundary Error "
        "vs Leisen-Reimer Extrapolated Boundary"
    )

    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":

    results = (
        run_boundary_convergence_study()
    )

    plot_boundary_convergence(
        results
    )
    plot_boundary_error(results)