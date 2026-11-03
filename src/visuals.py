from __future__ import annotations

from typing import Literal, Optional, Protocol, TYPE_CHECKING

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

if TYPE_CHECKING:
    from discrete_models import ConvergenceResults

class OptionSpecLike(Protocol):
    spot: float
    strike: float
    rate: float
    volatility: float
    maturity: float
    steps: int
    option_type: str
    exercise_style: str
    dividend_yield: float

DisplayMode = Literal["spot", "option", "combined"]

def add_spec_box(
    ax,
    spec : OptionSpecLike,
    model_name: str = "Jarrow-Rudd",
    u: Optional[float] = None,
    m: Optional[float] = None,
    d: Optional[float] = None,
    p: Optional[float] = None,
    p_u: Optional[float] = None,
    p_m: Optional[float] = None,
    p_d: Optional[float] = None,
    loc: str = "upper left",
    fontsize: int = 9,
    x: Optional[float] = None,
    y: Optional[float] = None,
):
    """
    Add a parameter summary box to an existing Matplotlib axis.
    If x and y are provided, they override loc and are interpreted
    in axis coordinates (ax.transAxes).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    spec : object
        Option specification object. Expected attributes:
        spot, strike, rate, volatility, maturity, steps,
        option_type, exercise_style, dividend_yield.
    model_name : str
        Name of the model to display.
    u, d, p : float, optional
        Model parameters to display if provided.
    loc : str
        Box location. Supported:
        'upper left', 'upper right', 'lower left', 'lower right'.
    fontsize : int
        Text font size.
    x : float, optional
    y : float, optional
    """
    option_label = str(spec.option_type).capitalize()
    exercise_label = str(spec.exercise_style).capitalize()

    dividend_yield = getattr(spec, "dividend_yield", 0.0)

    lines = [
        f"Model      : {model_name}",
        f"Option     : {option_label}",
        f"Exercise   : {exercise_label}",
        f"Spot       : {spec.spot:.4f}",
        f"Strike     : {spec.strike:.4f}",
        f"Rate       : {100.0 * spec.rate:.2f}%",
        f"Div. yield : {100.0 * dividend_yield:.2f}%",
        f"Volatility : {100.0 * spec.volatility:.2f}%",
        f"Maturity   : {spec.maturity:.4f}y",
        f"Steps      : {spec.steps}",
    ]

    # if u is not None:
    #     lines.append(f"u          : {u:.6f}")
    # if d is not None:
    #     lines.append(f"d          : {d:.6f}")
    # if p is not None:
    #     lines.append(f"p          : {p:.6f}")

    text = "\n".join(lines)

    anchor_map = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
    }

    if (x is None) != (y is None):
        raise ValueError("x and y must either both be provided or both be None")

    if x is not None and y is not None:
        x_pos: float = x
        y_pos: float = y
        ha = "left"
        va = "top"
    else:
        if loc not in anchor_map:
            raise ValueError("loc must be one of: 'upper left', 'upper right', 'lower left', 'lower right'")
        x_pos, y_pos, ha, va = anchor_map[loc]

    ax.text(
        x_pos,
        y_pos,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
        ),
        zorder=10,
    )

def _node_coordinates(
    step: int,
    up_moves: int,
    x_spacing: float = 1.6,
    y_spacing: float = 1.15,
) -> tuple[float, float]:
    """
    Map a binomial node (step, up_moves) to 2D plotting coordinates.
    """
    x = step * x_spacing
    y = (2 * up_moves - step) * y_spacing
    return float(x), float(y)


def _format_number(value: float, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{decimals}f}"


def _draw_node_box(
    ax,
    x: float,
    y: float,
    text: str,
    facecolor: str = "white",
    edgecolor: str = "black",
    linewidth: float = 1.0,
    width: float = 1.20,
    height: float = 0.72,
    fontsize: int = 8,
) -> None:
    patch = FancyBboxPatch(
        (x - width / 2.0, y - height / 2.0),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        family="monospace",
        zorder=4,
    )


def render_binomial_tree(
    spec: OptionSpecLike,
    spot_tree: np.ndarray,
    option_tree: Optional[np.ndarray] = None,
    exercise_tree: Optional[np.ndarray] = None,
    mode: DisplayMode = "combined",
    decimals: int = 4,
    title: Optional[str] = None,
    show_arrows: bool = True,
    figsize: tuple[float, float] | None = None,
    fontsize: int = 8,
    node_width: Optional[float] = None,
    node_height: Optional[float] = None,
    x_spacing: float = 1.6,
    y_spacing: float = 1.15,
):
    """
    Render a recombining binomial tree using Matplotlib.
    """
    if mode not in ("spot", "option", "combined"):
        raise ValueError("mode must be 'spot', 'option', or 'combined'")

    if mode in ("option", "combined") and option_tree is None:
        raise ValueError("option_tree is required when mode is 'option' or 'combined'")

    if spot_tree.ndim != 2 or spot_tree.shape[0] != spot_tree.shape[1]:
        raise ValueError("spot_tree must be a square 2D array, typically (N+1, N+1)")

    n_steps = spot_tree.shape[0] - 1

    if option_tree is not None and option_tree.shape != spot_tree.shape:
        raise ValueError("option_tree must have the same shape as spot_tree")

    if exercise_tree is not None and exercise_tree.shape != spot_tree.shape:
        raise ValueError("exercise_tree must have the same shape as spot_tree")

    if node_width is None:
        node_width = 1.10 if mode != "combined" else 1.45

    if node_height is None:
        node_height = 0.55 if mode != "combined" else 0.88

    if figsize is None:
        figsize = (
            max(12, 1.9 * (n_steps + 1)),
            max(7, 1.2 * (n_steps + 1)),
        )
    if node_width is None:
        node_width_value: float = 1.10 if mode != "combined" else 1.45
    else:
        node_width_value = node_width

    if node_height is None:
        node_height_value: float = 0.55 if mode != "combined" else 0.88
    else:
        node_height_value = node_height

    fig, ax = plt.subplots(figsize=figsize)

    if show_arrows:
        for i in range(n_steps):
            for j in range(i + 1):
                if np.isnan(spot_tree[i, j]):
                    continue

                x0, y0 = _node_coordinates(i, j, x_spacing=x_spacing, y_spacing=y_spacing)

                # Down move: same number of up moves
                x_down, y_down = _node_coordinates(i + 1, j, x_spacing=x_spacing, y_spacing=y_spacing)
                ax.plot(
                    [x0, x_down],
                    [y0, y_down],
                    linewidth=1.0,
                    color="#c43d3d",  # red
                    zorder=1,
                )

                # Up move: one extra up move
                x_up, y_up = _node_coordinates(i + 1, j + 1, x_spacing=x_spacing, y_spacing=y_spacing)
                ax.plot(
                    [x0, x_up],
                    [y0, y_up],
                    linewidth=1.0,
                    color="#2e9b4b",  # green
                    zorder=1,
                )

    for i in range(n_steps + 1):
        for j in range(i + 1):
            s_ij = spot_tree[i, j]
            if np.isnan(s_ij):
                continue

            x, y = _node_coordinates(i, j, x_spacing=x_spacing, y_spacing=y_spacing)

            if mode == "spot":
                text = f"S={_format_number(s_ij, decimals)}"
                facecolor = "#fff2a8"
                edgecolor = "black"

            elif mode == "option":
                v_ij = option_tree[i, j]
                text = f"V={_format_number(v_ij, decimals)}"
                facecolor = "#bfe7ff"
                edgecolor = "black"

            else:
                v_ij = option_tree[i, j]
                text = (
                    f"S={_format_number(s_ij, decimals)}\n"
                    f"V={_format_number(v_ij, decimals)}"
                )
                facecolor = "white"
                edgecolor = "black"

            linewidth = 1.0
            edgecolor = "black"

            # Terminal nodes first
            if i == n_steps:
                edgecolor = "#2b6cff"  # blue
                linewidth = 2.0

            # Early exercise overrides only before maturity
            elif exercise_tree is not None and bool(exercise_tree[i, j]):
                edgecolor = "#cc2f2f"  # red
                linewidth = 2.2

            _draw_node_box(
                ax=ax,
                x=x,
                y=y,
                text=text,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                width=node_width_value,
                height=node_height_value,
                fontsize=fontsize,
            )

    ax.set_aspect("equal")
    ax.axis("off")

    if title is None:
        if mode == "spot":
            title_value = "Binomial Spot Tree"
        elif mode == "option":
            title_value = "Binomial Option Value Tree"
        else:
            title_value = "Binomial Spot / Option Tree"
    else: title_value = title

    if spec.exercise_style == "American":
        s = 'Blue frames: maturity nodes · Red frames: early exercise nodes'
    else:
        s = 'Blue frames: maturity nodes'

    ax.set_title(title_value)
    ax.text(
        0.5,
        0.98,
        s,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    return fig, ax

def _format_log_steps_axis(ax) -> None:
    ax.set_xscale("log")

    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, pos: f"{x:g}"
        )
    )

def _trinomial_node_coordinates(
    step: int,
    j: int,
    x_spacing: float = 1.8,
    y_spacing: float = 1.15,
) -> tuple[float, float]:
    """
    Map a trinomial node (step, j) to 2D plotting coordinates.

    At time step i, valid node indices are j = 0, ..., 2*i.
    The vertical state is k = i - j, so the center node has k = 0.
    """
    x = step * x_spacing
    state = step - j
    y = state * y_spacing
    return float(x), float(y)

def render_trinomial_tree(
    spec: OptionSpecLike,
    spot_tree: np.ndarray,
    option_tree: Optional[np.ndarray] = None,
    exercise_tree: Optional[np.ndarray] = None,
    mode: DisplayMode = "combined",
    decimals: int = 4,
    title: Optional[str] = None,
    show_arrows: bool = True,
    figsize: tuple[float, float] | None = None,
    fontsize: int = 8,
    node_width: Optional[float] = None,
    node_height: Optional[float] = None,
    x_spacing: float = 1.8,
    y_spacing: float = 1.15,
):
    """
        Render a recombining trinomial tree using Matplotlib.
    """
    if spot_tree.ndim != 2:
        raise ValueError("spot_tree must be a 2D array")
    n_steps = spot_tree.shape[0] - 1

    expected_width = 2 * n_steps + 1
    if spot_tree.shape[1] != expected_width:
        raise ValueError(
            "spot_tree must have shape (n+1, 2*n+1) for a trinomial tree"
        )

    if mode in ("option", "combined") and option_tree is None:
        raise ValueError(
            "option_tree is required for option or combined mode."
        )

    if option_tree is not None and option_tree.shape != spot_tree.shape:
        raise ValueError(
            "option_tree must have the same shape as spot_tree."
        )

    if exercise_tree is not None and exercise_tree.shape != spot_tree.shape:
        raise ValueError(
            "exercise_tree must have the same shape as spot_tree."
        )

    node_width_value = (
        node_width if node_width is not None else 1.25
    )

    node_height_value = (
        node_height if node_height is not None else 0.55
    )

    if figsize is None:
        figsize = (
            max(8.0, 2.0 + n_steps * x_spacing),
            max(5.0, 2.0 + 2 * n_steps * y_spacing),
        )

    fig, ax = plt.subplots(figsize=figsize)

    if show_arrows:
        for i in range(n_steps):
            for j in range(2 * i + 1):
                if np.isnan(spot_tree[i, j]):
                    continue

                x0, y0 = _trinomial_node_coordinates(
                    i, j, x_spacing=x_spacing, y_spacing=y_spacing
                )

                # Up child: (i+1, j)
                x_up, y_up = _trinomial_node_coordinates(
                    i + 1, j, x_spacing=x_spacing, y_spacing=y_spacing
                )
                ax.plot([x0, x_up], [y0, y_up], linewidth=1.0, color="#2e9b4b", zorder=1)

                # Middle child: (i+1, j+1)
                x_mid, y_mid = _trinomial_node_coordinates(
                    i + 1, j + 1, x_spacing=x_spacing, y_spacing=y_spacing
                )
                ax.plot([x0, x_mid], [y0, y_mid], linewidth=1.0, color="#d8a600", zorder=1)

                # Down child: (i+1, j+2)
                x_down, y_down = _trinomial_node_coordinates(
                    i + 1, j + 2, x_spacing=x_spacing, y_spacing=y_spacing
                )
                ax.plot([x0, x_down], [y0, y_down], linewidth=1.0, color="#c43d3d", zorder=1)

    for i in range(n_steps + 1):
        for j in range(2 * i + 1):
            s_ij = spot_tree[i, j]
            if np.isnan(s_ij):
                continue

            x, y = _trinomial_node_coordinates(i, j, x_spacing=x_spacing, y_spacing=y_spacing)

            if mode == "spot":
                text = f"S={_format_number(s_ij, decimals)}"
                facecolor = "#fff2a8"
                edgecolor = "black"

            elif mode == "option":
                v_ij = option_tree[i, j]
                text = f"V={_format_number(v_ij, decimals)}"
                facecolor = "#bfe7ff"
                edgecolor = "black"

            else:
                v_ij = option_tree[i, j]
                text = (
                    f"S={_format_number(s_ij, decimals)}\n"
                    f"V={_format_number(v_ij, decimals)}"
                )
                facecolor = "white"
                edgecolor = "black"

            linewidth = 1.0
            edgecolor = "black"

            # Terminal nodes first
            if i == n_steps:
                edgecolor = "#2b6cff"  # blue
                linewidth = 2.0

            # Early exercise overrides only before maturity
            elif exercise_tree is not None and bool(exercise_tree[i, j]):
                edgecolor = "#cc2f2f"  # red
                linewidth = 2.2

            _draw_node_box(
                ax=ax,
                x=x,
                y=y,
                text=text,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                width=node_width_value,
                height=node_height_value,
                fontsize=fontsize,
            )
    if title is None:
        if mode == "spot":
            title_value = "Binomial Spot Tree"
        elif mode == "option":
            title_value = "Binomial Option Value Tree"
        else:
            title_value = "Binomial Spot / Option Tree"
    else: title_value = title

    if spec.exercise_style == "American":
        s = 'Blue frames: maturity nodes · Red frames: early exercise nodes'
    else:
        s = 'Blue frames: maturity nodes'

    ax.axis("off")
    ax.set_title(title_value)

    ax.text(
        0.5,
        0.98,
        s,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )

    return fig, ax

def plot_price_convergence(
    results: "ConvergenceResults",
    results_trino: "ConvergenceResults | None" = None,
    benchmarks: dict[str, float] | None = None,
) -> Figure:

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.plot(
        results.steps,
        results.crr_averaged,
        marker="o",
        label="CRR averaged",
    )

    ax.plot(
        results.steps,
        results.jarrow_rudd,
        marker="o",
        label="Jarrow-Rudd",
    )

    ax.plot(
        results.steps,
        results.tian93,
        marker="o",
        label="Tian 1993",
    )

    ax.plot(
        results.steps,
        results.leisen_reimer,
        marker="o",
        label="Leisen-Reimer",
    )

    ax.plot(
        results.steps,
        results.tian99,
        marker="o",
        label="Tian 1999",
    )

    ax.plot(
        results.steps,
        results_trino.kamrad_ritchken,
        marker="o",
        label="KR-Trino",
    )

    benchmark_styles = {
        "BAW": {"linestyle": "--", "linewidth": 1.6},
        "Ju-Zhong": {"linestyle": ":", "linewidth": 2.0},
        "BMS": {"linestyle": "-.", "linewidth": 1.5},
    }

    if benchmarks:
        for label, value in benchmarks.items():
            style = benchmark_styles.get(
                label,
                {"linestyle": "--", "linewidth": 1.5},
            )

            ax.axhline(
                y=value,
                label=label,
                **style,
            )

    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Option price")
    ax.set_title("Binomial price convergence")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, pos: f"{x:g}"
        )
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    return fig

def plot_relative_error(
    results: "ConvergenceResults",
    benchmark: float,
    results_trino: "ConvergenceResults | None" = None,
) -> Figure:

    if benchmark == 0.0:
        raise ValueError(
            "Relative error is undefined for a zero benchmark."
        )

    steps = np.asarray(results.steps)

    denominator = abs(benchmark)

    crr_error = (
        np.abs(np.asarray(results.crr_averaged) - benchmark)
        / denominator
        * 100.0
    )

    jr_error = (
        np.abs(np.asarray(results.jarrow_rudd) - benchmark)
        / denominator
        * 100.0
    )

    tian93_error = (
        np.abs(np.asarray(results.tian93) - benchmark)
        / denominator
        * 100.0
    )

    lr_error = (
        np.abs(np.asarray(results.leisen_reimer) - benchmark)
        / denominator
        * 100.0
    )

    tian99_error = (
        np.abs(np.asarray(results.tian99) - benchmark)
        / denominator
    )

    kr_error = (
            np.abs(np.asarray(results_trino.kamrad_ritchken) - benchmark)
            / denominator
    )

    crr_error = np.where(crr_error > 0.0, crr_error, np.nan)
    jr_error = np.where(jr_error > 0.0, jr_error, np.nan)
    tian93_error = np.where(tian93_error > 0.0, tian93_error, np.nan)
    lr_error = np.where(lr_error > 0.0, lr_error, np.nan)
    tian99_error = np.where(tian99_error > 0.0, tian99_error, np.nan)
    kr_error = np.where(kr_error > 0.0, kr_error, np.nan)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.plot(
        steps,
        crr_error,
        marker="o",
        label="CRR averaged",
    )

    ax.plot(
        steps,
        jr_error,
        marker="o",
        label="Jarrow-Rudd",
    )

    ax.plot(
        steps,
        tian93_error,
        marker="o",
        label="Tian 1993",
    )

    ax.plot(
        results.steps,
        lr_error,
        marker="o",
        label="Leisen-Reimer",
    )

    ax.plot(
        results.steps,
        tian99_error,
        marker="o",
        label="Tian 1999",
    )

    ax.plot(
        results_trino.steps,
        kr_error,
        marker="o",
        label="KR-Trino",
    )

    _format_log_steps_axis(ax)

    ax.set_yscale("log")

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda y, pos: f"{y:g}%"
        )
    )

    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Relative pricing error")
    ax.set_title("Relative convergence error")

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    return fig

def plot_absolute_error(
    results: "ConvergenceResults",
    benchmark: float,
    results_trino: "ConvergenceResults | None" = None,
) -> Figure:

    steps = np.asarray(results.steps)

    crr_error = np.abs(
        np.asarray(results.crr_averaged) - benchmark
    )

    jr_error = np.abs(
        np.asarray(results.jarrow_rudd) - benchmark
    )

    tian93_error = np.abs(
        np.asarray(results.tian93) - benchmark
    )

    lr_error = np.abs(
        np.asarray(results.leisen_reimer) - benchmark
    )

    tian99_error = np.abs(
        np.asarray(results.tian99) - benchmark
    )

    kr_error = np.abs(np.asarray(results_trino.kamrad_ritchken) - benchmark)

    # Log scale cannot display exact zero.
    crr_error = np.where(crr_error > 0.0, crr_error, np.nan)
    jr_error = np.where(jr_error > 0.0, jr_error, np.nan)
    tian93_error = np.where(tian93_error > 0.0, tian93_error, np.nan)
    lr_error = np.where(lr_error > 0.0, lr_error, np.nan)
    tian99_error = np.where(tian99_error > 0.0, tian99_error, np.nan)
    kr_error = np.where(kr_error > 0.0, kr_error, np.nan)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.plot(
        steps,
        crr_error,
        marker="o",
        label="CRR averaged",
    )

    ax.plot(
        steps,
        jr_error,
        marker="o",
        label="Jarrow-Rudd",
    )

    ax.plot(
        steps,
        tian93_error,
        marker="o",
        label="Tian 1993",
    )

    ax.plot(
        results.steps,
        lr_error,
        marker="o",
        label="Leisen-Reimer",
    )

    ax.plot(
        results.steps,
        tian99_error,
        marker="o",
        label="Tian 1999",
    )

    ax.plot(
        results_trino.steps,
        kr_error,
        marker="o",
        label="KR-Trino",
    )

    _format_log_steps_axis(ax)

    ax.set_yscale("log")

    # Avoid MathText formatting on the logarithmic y-axis too.
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda y, pos: f"{y:.0e}"
        )
    )

    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Absolute pricing error")
    ax.set_title("Absolute convergence error")

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    return fig

def plot_signed_error(
    results: "ConvergenceResults",
    benchmark: float,
    results_trino: "ConvergenceResults | None" = None,
) -> Figure:

    steps = np.asarray(results.steps)

    crr_error = (
        np.asarray(results.crr_averaged) - benchmark
    )

    jr_error = (
        np.asarray(results.jarrow_rudd) - benchmark
    )

    tian93_error = (
        np.asarray(results.tian93) - benchmark
    )

    lr_error = (
        np.asarray(results.leisen_reimer) - benchmark
    )

    tian99_error = (
        np.asarray(results.tian99) - benchmark
    )

    kr_error = np.asarray(results_trino.kamrad_ritchken) - benchmark

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.plot(
        steps,
        crr_error,
        marker="o",
        label="CRR averaged",
    )

    ax.plot(
        steps,
        jr_error,
        marker="o",
        label="Jarrow-Rudd",
    )

    ax.plot(
        steps,
        tian93_error,
        marker="o",
        label="Tian 1993",
    )

    ax.plot(
        results.steps,
        lr_error,
        marker="o",
        label="Leisen-Reimer",
    )

    ax.plot(
        results.steps,
        tian99_error,
        marker="o",
        label="Tian 1999",
    )

    ax.plot(
        results.steps,
        kr_error,
        marker="o",
        label="KR-Trino",
    )

    _format_log_steps_axis(ax)

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        label="Benchmark",
    )

    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Signed pricing error")
    ax.set_title("Signed convergence error")

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    return fig

def plot_crr_smoothing(
    results: "ConvergenceResults",
    benchmark: float,
) -> Figure:

    steps = np.asarray(results.steps)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.plot(
        steps,
        results.crr,
        marker="o",
        label="CRR raw",
    )

    ax.plot(
        steps,
        results.crr_averaged,
        marker="o",
        label="CRR averaged",
    )

    ax.axhline(
        benchmark,
        linestyle="--",
        linewidth=1.0,
        label="Black-Scholes",
    )

    _format_log_steps_axis(ax)

    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Option price")
    ax.set_title("CRR odd/even smoothing")

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    return fig

def plot_convergence_performance(
    threaded_time: float,
    unthreaded_time: float,
) -> Figure:

    labels = [
        "Threaded",
        "Unthreaded",
    ]

    times = [
        threaded_time,
        unthreaded_time,
    ]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    bars = ax.bar(
        labels,
        times,
    )

    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Convergence computation performance")

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    for bar, value in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f} s",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    return fig