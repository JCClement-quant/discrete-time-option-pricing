from linker_cpp import discrete_models
from time import perf_counter

market = discrete_models.MarketData(
    spot=100.0,
    rate=0.05,
    dividend_yield=0.0,
    volatility=0.20,
)

steps_grid = [5, 10, 15, 25, 50, 100, 250, 1000, 2500, 5000, 10000]

call = discrete_models.PayoffCall(100);
option = discrete_models.VanillaOption(call, 1, discrete_models.ExerciseStyle.European)

t0 = perf_counter()
results = discrete_models.Binomial_Convergence(
    option,
    market,
    steps_grid,
)
threaded_time = perf_counter() - t0

t0 = perf_counter()
results_unthreaded = discrete_models.Binomial_Convergence_Unthreaded(
    option,
    market,
    steps_grid,
)
unthreaded_time = perf_counter() - t0

speedup = unthreaded_time / threaded_time
print(speedup)