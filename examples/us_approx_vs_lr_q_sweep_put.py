from src.pricers import OptionSpec
from src.pricers import BAW_put, BMS_at_spot, BAW_quadratic, BAW_put_critical_spot, JZ_Pricer
import numpy as np
import matplotlib.pyplot as plt
from linker_cpp import discrete_models
from scipy.optimize import brentq

def run_us_approx_vs_lr_q_sweep():
    #q_grid = np.linspace(0.01, 0.6, num=100)
    q_grid = [0.08,0.23,0.35,0.50,0.60] # For N = 10001

    lr_prices = []
    bms_prices = []
    baw_prices = []
    jz_prices = []

    abs_errors_baw_bms = []
    signed_errors_baw_bms = []
    abs_errors_jz_bms = []
    signed_errors_jz_bms = []
    abs_errors_lr_bms = []
    signed_errors_lr_bms = []

    delta_baw_lr = []
    delta_jz_lr = []
    delta_jz_baw = []

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

        #config = discrete_models.TreeConfig(int(1001))  # C++
        config = discrete_models.TreeConfig(int(1001))  # C++

        # lr = lr_price(spec)
        lr = discrete_models.LR_cpp_price(option, market, config)
        baw = BAW_put(spec, spec.spot)
        jz = JZ_Pricer(spec, spec.spot, -1)
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
        lower = max(floor, 0.8 * S_star_baw)

        upper = min(spec.strike, 1.2 * S_star_baw)

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
        jz_prices.append(jz)
        baw_prices.append(baw)
        bms_prices.append(bms)

        abs_errors_baw_bms.append(abs(baw - bms))
        signed_errors_baw_bms.append(baw - bms)
        abs_errors_jz_bms.append(abs(jz - bms))
        signed_errors_jz_bms.append(jz - bms)
        abs_errors_lr_bms.append(abs(lr - bms))
        signed_errors_lr_bms.append(lr - bms)

        delta_baw_lr.append(baw - lr)
        delta_jz_lr.append(jz - lr)
        delta_jz_baw.append(jz - baw)

        if abs(baw - lr) > 0.00001:
            print(
                #f"steps={tango:.3f} || "
                f"q={q:.3f} | "
                f"LR={lr:.6f} | "
                f"BAW={baw:.6f} | "
                f"Dbaw_lr={baw - lr:.6f} | "
                f"Djz_lr={jz - lr:.6f} | "
                f"Djz_baw={jz - baw:.6f} | "
            )

    # Plot 1: American Put Price: BAW-LR vs JZ-LR
    plt.figure(figsize=(10, 6))
    plt.plot(q_grid, delta_baw_lr, label="Diff BAW-LR")
    plt.plot(q_grid, delta_jz_lr, label="Diff JZ-LR")
    plt.xlabel("Dividend yield q")
    plt.ylabel("Premium difference")
    plt.title("American Put Price: BAW-LR vs JZ-LR")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 2: American Put Price: Diff JZ-BAW
    plt.figure(figsize=(10, 6))
    plt.plot(q_grid, delta_jz_baw, label="Diff JZ-BAW")
    plt.xlabel("Dividend yield q")
    plt.ylabel("Premium difference")
    plt.title("American Put Price: Diff JZ-BAW")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
        run_us_approx_vs_lr_q_sweep()