from src.pricers import OptionSpec, lr_price, BAW_call, BMS_at_spot, BAW_quadratic, BAW_call_critical_spot
import numpy as np
import matplotlib.pyplot as plt
from linker_cpp import discrete_models
from scipy.optimize import brentq

def run_baw_vs_lr_q_sweep():
    q_grid = np.linspace(0.01, 0.6, num=100)

    lr_prices = []
    bms_prices = []
    baw_prices = []

    abs_errors = []
    signed_errors = []

    delta_lr_bms = []
    delta_baw_bms = []

    lr_boundaries = []
    baw_boundaries = []

    for q in q_grid:
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

        payoff = discrete_models.PayoffCall(spec.strike)
        cpp_exercise_style = discrete_models.ExerciseStyle.American
        option = discrete_models.VanillaOption(payoff,
                                               spec.maturity,
                                               cpp_exercise_style
                                               )
        market = discrete_models.MarketData(spec.spot,
                                            spec.rate,
                                            spec.dividend_yield,
                                            spec.volatility
                                            )

        config = discrete_models.TreeConfig(int(1001))  # C++

        # lr = lr_price(spec)
        lr = discrete_models.LR_cpp_price(option, market, config)
        baw = BAW_call(spec, spec.spot)
        bms = BMS_at_spot(spec, spec.spot)[0]

        # BAW boundary
        Q2 = BAW_quadratic(spec)[1]
        S_star_baw = BAW_call_critical_spot(spec, Q2)

        # LR boundary, q fixed, moves S
        def lr_gap_at_spot(s):

            bumped_market = discrete_models.MarketData(
                float(s),
                spec.rate,
                spec.dividend_yield,
                spec.volatility,
            )

            return discrete_models.LR_root(
                option,
                bumped_market,
                config,
            )

        # Find a bracket
        lower = max(spec.strike, 0.90 * S_star_baw)
        upper = 1.10 * S_star_baw

        f_lower = lr_gap_at_spot(lower)
        f_upper = lr_gap_at_spot(upper)

        while f_lower * f_upper > 0.0:
            lower = max(spec.strike, lower * 0.95)
            upper *= 1.5
            f_lower = lr_gap_at_spot(lower)
            f_upper = lr_gap_at_spot(upper)

            if upper > 1.0e8:
                raise RuntimeError(
                    f"Unable to bracket LR boundary for q={q}"
                )

        S_star_lr = brentq(
            lr_gap_at_spot,
            lower,
            upper,
            xtol=1e-4,
            rtol=1e-8,
            maxiter=50,
        )

        lr_prices.append(lr)
        baw_prices.append(baw)
        bms_prices.append(bms)

        abs_errors.append(abs(baw - lr))
        signed_errors.append(baw - lr)

        delta_lr_bms.append(lr - bms)
        delta_baw_bms.append(baw - bms)

        baw_boundaries.append(S_star_baw)
        lr_boundaries.append(S_star_lr)

        if abs(baw - lr) > 0.02:
            print(
                #f"steps={tango:.3f} || "
                f"q={q:.3f} | "
                f"LR={lr:.6f} | "
                f"BAW={baw:.6f} | "
                f"diff={baw - lr:.6f} | "
                f"S*_LR={S_star_lr:.6f} | "
                f"S*_BAW={S_star_baw:.6f} | "
                f"dS*={S_star_baw - S_star_lr:.6f}"
            )

    # Plot 1: prices
    plt.figure(figsize=(10, 6))
    plt.plot(q_grid, lr_prices, label="LR")
    plt.plot(q_grid, baw_prices, label="BAW")
    plt.xlabel("Dividend yield q")
    plt.ylabel("Option price")
    plt.title("American Call Price: Leisen-Reimer vs BAW")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 2: absolute error
    plt.figure(figsize=(10, 6))
    plt.plot(q_grid, abs_errors, label="|BAW - Leisen-Reimer|")
    plt.xlabel("Dividend yield q")
    plt.ylabel("Absolute error")
    plt.title("Absolute Pricing Error of BAW vs Leisen-Reimer")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 3: signed error
    plt.figure(figsize=(10, 6))
    plt.plot(q_grid, signed_errors, label="BAW - Leisen-Reimer")
    plt.xlabel("Dividend yield q")
    plt.ylabel("Signed pricing error")
    plt.title("Signed Pricing Error of BAW vs Leisen-Reimer")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 4: early exercice premium
    plt.figure(figsize=(10, 6))
    plt.plot(q_grid, delta_lr_bms, label="Leisen-Reimer - BMS")
    plt.plot(q_grid, delta_baw_bms, label="BAW - BMS")
    plt.xlabel("Dividend yield q")
    plt.ylabel("Option price")
    plt.title("American Call Price: Leisen-Reimer vs BAW early-exercise premia")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 5: early exercice premium
    plt.figure(figsize=(10, 6))

    plt.plot(
        q_grid,
        lr_boundaries,
        label="Leisen-Reimer S*",
    )

    plt.plot(
        q_grid,
        baw_boundaries,
        label="BAW S*",
    )

    plt.xlabel("Dividend yield q")
    plt.ylabel("Critical spot S*")
    plt.title("American Call Exercise Frontier: LR vs BAW")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 6 : Delta_V(q) over Delta_S*(q)
    boundary_errors = (
            np.asarray(baw_boundaries)
            - np.asarray(lr_boundaries)
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        q_grid,
        boundary_errors,
        label="BAW S* - LR S*",
    )

    plt.axhline(0.0)

    plt.xlabel("Dividend yield q")
    plt.ylabel("Boundary error")
    plt.title("BAW Exercise-Boundary Error vs Leisen-Reimer")
    plt.legend()
    plt.grid(True)

if __name__ == "__main__":
    run_baw_vs_lr_q_sweep()