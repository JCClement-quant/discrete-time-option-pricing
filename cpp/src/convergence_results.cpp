#include "convergence_results.h"
#include <thread>
#include <stop_token>
#include <mutex>
#include <condition_variable>
#include <queue>

ConvergenceResults binomial_convergence(const discrete_models::VanillaOption &option,
    const discrete_models::MarketData &market_data,
    const std::vector<int> &steps_grid
)
{
    ConvergenceResults results;
    results.steps = steps_grid;
    results.crr.reserve(steps_grid.size());
    results.crr_averaged.reserve(steps_grid.size());
    results.jarrow_rudd.reserve(steps_grid.size());
    results.tian93.reserve(steps_grid.size());

    for (const int steps : steps_grid) {

        const discrete_models::TreeConfig config{steps};
        const discrete_models::TreeConfig config_shifted{steps +1 };

        double crr_n{};
        double crr_n_shifted{};
        double jr_n{};
        double tian93_n{};

        std::jthread worker_crr(
            [&] {
                crr_n = discrete_models::CRR_price(option, market_data, config);
            }
        );
        std::jthread worker_crr_averaged(
            [&] {
                crr_n_shifted = discrete_models::CRR_price(option, market_data, config_shifted);
            }
        );
        std::jthread worker_jr(
            [&] {
                jr_n = discrete_models::JR_price(option, market_data, config);
            }
        );
        std::jthread worker_tian93(
            [&] {
                tian93_n = discrete_models::TIAN93_price(option, market_data, config);
            }
        );

        worker_crr.join();
        worker_crr_averaged.join();
        worker_jr.join();
        worker_tian93.join();

        results.crr.push_back(crr_n);
        results.crr_averaged.push_back(0.5*(crr_n + crr_n_shifted));
        results.jarrow_rudd.push_back(jr_n);
        results.tian93.push_back(tian93_n);
    }
    return results;
}

ConvergenceResults binomial_convergence_unthreaded(const discrete_models::VanillaOption &option,
    const discrete_models::MarketData &market_data,
    const std::vector<int> &steps_grid
)
{
    ConvergenceResults results;
    results.steps = steps_grid;
    results.crr.reserve(steps_grid.size());
    results.crr_averaged.reserve(steps_grid.size());
    results.jarrow_rudd.reserve(steps_grid.size());
    results.tian93.reserve(steps_grid.size());

    for (const int steps : steps_grid) {

        const discrete_models::TreeConfig config{steps};
        const discrete_models::TreeConfig config_shifted{steps +1 };

        double crr_n = discrete_models::CRR_price(option, market_data, config);
        double crr_n_shifted = discrete_models::CRR_price(option, market_data, config_shifted);
        double jr_n = discrete_models::JR_price(option, market_data, config);
        double tian93_n = discrete_models::TIAN93_price(option, market_data, config);

        results.crr.push_back(crr_n);
        results.crr_averaged.push_back(0.5*(crr_n + crr_n_shifted));
        results.jarrow_rudd.push_back(jr_n);
        results.tian93.push_back(tian93_n);
    }
    return results;
}

