from __future__ import annotations

from typing import Literal, Optional, Protocol

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

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
    d: Optional[float] = None,
    p: Optional[float] = None,
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

    if u is not None:
        lines.append(f"u          : {u:.6f}")
    if d is not None:
        lines.append(f"d          : {d:.6f}")
    if p is not None:
        lines.append(f"p          : {p:.6f}")

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
    figsize: Optional[tuple[float, float]] = None,
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

    if spec.exercise_style == "american":
        s = 'Blue frames: maturity nodes · Red frames: early exercise nodes'
    else:
        s = 'Blue frames: early exercise nodes'

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
