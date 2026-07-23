from linker_cpp import discrete_models
from src.pricers import OptionSpec, crr_price
from time import perf_counter

for N in [1000, 2500, 5000, 10000]:

    market = discrete_models.MarketData(
        spot=100.0,
        rate=0.05,
        dividend_yield=0.0,
        volatility=0.20,
    )

    config = discrete_models.TreeConfig(
        steps=N,
    )

    put = discrete_models.PayoffPut(100);
    option = discrete_models.VanillaOption(put, 1, discrete_models.ExerciseStyle.American)

    cpp_crr_price = discrete_models.CRR_cpp_price(
        option,
        market,
        config,
    )

    spec_crr_put = OptionSpec(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=N,
        option_type="Put",
        exercise_style="American",
        model="CRR",
        dividend_yield=0.0,
        )

    start = perf_counter()
    python_price = crr_price(spec_crr_put)
    python_elapsed = perf_counter() - start

    start = perf_counter()
    cpp_price = discrete_models.CRR_cpp_price(
        option,
        market,
        config,
    )
    cpp_elapsed = perf_counter() - start

    print(f"N: {N:.0f} steps")
    print(f"Python:  {python_elapsed:.6f} s")
    print(f"C++:     {cpp_elapsed:.6f} s")
    print(f"Speedup: {python_elapsed / cpp_elapsed:.1f}x")