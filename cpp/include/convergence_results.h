#include "binomial_pricer.h"
#include "trinomial_pricer.h"
#include <vector>

struct ConvergenceResults {
    std::vector<int> steps;
    std::vector<double> crr;
    std::vector<double> crr_averaged;
    std::vector<double> jarrow_rudd;
    std::vector<double> tian93;
    std::vector<double> leisen_reimer;
    std::vector<double> tian99;
    std::vector<double> kamrad_ritchken;
};

[[nodiscard]]
ConvergenceResults binomial_convergence(
    const discrete_models::VanillaOption& option,
    const discrete_models::MarketData& market_data,
    const std::vector<int>& steps_grid
);

[[nodiscard]]
ConvergenceResults binomial_convergence_unthreaded(
    const discrete_models::VanillaOption& option,
    const discrete_models::MarketData& market_data,
    const std::vector<int>& steps_grid
);

[[nodiscard]]
ConvergenceResults trinomial_convergence(
    const discrete_models::VanillaOption& option,
    const discrete_models::MarketData& market_data,
    const std::vector<int>& steps_grid,
    const discrete_models::KRconfig& krconfig = {}
);

[[nodiscard]]
ConvergenceResults trinomial_convergence_unthreaded(
    const discrete_models::VanillaOption& option,
    const discrete_models::MarketData& market_data,
    const std::vector<int>& steps_grid,
    const discrete_models::KRconfig& krconfig = {}
);

