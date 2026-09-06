from __future__ import annotations
from src.pricers import BAW_quadratic, BAW_put_critical_spot, BAW_put
from src.pricers import OptionSpec
from examples.baw_lr_boundary_convergence_put import lr_put_critical_spot
import numpy as np
from linker_cpp import discrete_models

q_local = np.arange(0.050, 0.0701, 0.002)

rows = []

for q in q_local:

    spec = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.20,
        steps=1001,
        option_type="Put",
        exercise_style="American",
        model="Leisen-Reimer",
        dividend_yield=float(q),
    )

    payoff = discrete_models.PayoffPut(spec.strike)

    option = discrete_models.VanillaOption(
        payoff,
        spec.maturity,
        discrete_models.ExerciseStyle.American,
    )

    market = discrete_models.MarketData(
        spec.spot,
        spec.rate,
        spec.dividend_yield,
        spec.volatility,
    )

    # ---------------------------------------------------------
    # Pricing
    # ---------------------------------------------------------

    config_price = discrete_models.TreeConfig(10001)

    lr_price = discrete_models.LR_cpp_price(
        option,
        market,
        config_price,
    )

    baw_price = BAW_put(
        spec,
        spec.spot,
    )

    pricing_error = baw_price - lr_price

    # ---------------------------------------------------------
    # BAW boundary
    # ---------------------------------------------------------

    Q1 = BAW_quadratic(spec)[0]

    baw_boundary = BAW_put_critical_spot(
        spec,
        Q1,
    )

    # ---------------------------------------------------------
    # LR boundary convergence
    # ---------------------------------------------------------

    lr_boundaries_N = []
    boundary_guess = baw_boundary

    for steps in [1001, 3001, 5001, 10001]:

        config = discrete_models.TreeConfig(steps)

        lr_boundary = lr_put_critical_spot(
            spec,
            option,
            config,
            initial_guess=boundary_guess,
        )

        lr_boundaries_N.append(lr_boundary)
        boundary_guess = lr_boundary

    steps_grid = np.array(
        [1001, 3001, 5001, 10001],
        dtype=float,
    )

    x = 1.0 / np.sqrt(steps_grid)

    slope, lr_boundary_inf = np.polyfit(
        x,
        np.asarray(lr_boundaries_N),
        1,
    )

    boundary_error = (
        baw_boundary - lr_boundary_inf
    )

    relative_boundary_error = (
        boundary_error / lr_boundary_inf
    )

    rows.append(
        (
            q,
            baw_price,
            lr_price,
            pricing_error,
            baw_boundary,
            lr_boundary_inf,
            boundary_error,
            100.0 * relative_boundary_error,
        )
    )

    print(
        f"{'q':>6} "
        f"{'BAW':>10} "
        f"{'LR':>10} "
        f"{'dV':>10} "
        f"{'S*_BAW':>11} "
        f"{'S*_LRinf':>11} "
        f"{'dS*':>10} "
        f"{'dS* %':>10}"
    )

    for row in rows:
        (
            q,
            baw,
            lr,
            dv,
            sbaw,
            slr,
            ds,
            ds_pct,
    ) = row

        print(
            f"{q:6.3f} "
            f"{baw:10.6f} "
            f"{lr:10.6f} "
            f"{dv:+10.6f} "
            f"{sbaw:11.6f} "
            f"{slr:11.6f} "
            f"{ds:+10.6f} "
            f"{ds_pct:+9.3f}%"
        )