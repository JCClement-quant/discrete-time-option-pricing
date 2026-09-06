from src.pricers import OptionSpec, lr_price, BAW_put, BMS_at_spot, BAW_quadratic, BAW_put_critical_spot
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
            option_type="Put",
            exercise_style="American",
            model="Leisen-Reimer",
            dividend_yield=float(q),
        )

        payoff = discrete_models.PayoffPut(spec.strike)
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
        baw = BAW_put(spec, spec.spot)
        bms = BMS_at_spot(spec, spec.spot)[1]

        # BAW boundary
        Q1 = BAW_quadratic(spec)[0]
        S_star_baw = BAW_put_critical_spot(spec, Q1)

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

        floor = spec.strike * 1.0e-8

        # Find a bracket
        lower = max(floor,0.8 * S_star_baw)

        upper = min(spec.strike,1.2 * S_star_baw)

        f_lower = lr_gap_at_spot(lower)
        f_upper = lr_gap_at_spot(upper)

        while f_lower > 0.0 and lower > floor:
            lower = max(
                floor,
                0.8 * lower,
            )

            f_lower = lr_gap_at_spot(lower)

        while f_upper < 0.0 and upper < spec.strike:
            upper = min(
                spec.strike,
                1.2 * upper,
            )

            f_upper = lr_gap_at_spot(upper)

        if f_lower > 0.0 or f_upper < 0.0:
            raise RuntimeError(
                "Unable to bracket LR put exercise boundary. "
                f"q={spec.dividend_yield:.6f}, "
                f"lower={lower:.6f}, "
                f"f(lower)={f_lower:.6e}, "
                f"upper={upper:.6f}, "
                f"f(upper)={f_upper:.6e}"
            )

        S_star_lr = brentq(
            lr_gap_at_spot,
            lower,
            upper,
            xtol=1e-4,
            rtol=1e-8,
            maxiter=50,
        )

        assert 0.0 < S_star_lr < spec.strike

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
    plt.title("American Put Price: Leisen-Reimer vs BAW")
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
    plt.title("American Put Price: Leisen-Reimer vs BAW early-exercise premia")
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
    plt.title("American Put Exercise Frontier: LR vs BAW")
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