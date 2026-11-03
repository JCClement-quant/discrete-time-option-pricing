#include<trinomial_pricer.h>
#include <utility>
#include <cmath>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <vector>
#include <ranges>
#include <cassert>


namespace discrete_models{
    double Kamrad_Ritchken(const VanillaOption &option, const MarketData &market,
        const TreeConfig &treeconfig, const KRconfig &krconfig) {
        if (market.spot <= 0.0) {
            throw std::invalid_argument(
                "Spot must be strictly positive."
            );
        }

        if (option.maturity() <= 0.0) {
            throw std::invalid_argument(
                "Maturity must be strictly positive."
            );
        }

        if (market.volatility <= 0.0) {
            throw std::invalid_argument(
                "Volatility must be strictly positive."
            );
        }

        if (treeconfig.steps <= 0) {
            throw std::invalid_argument(
                "The number of steps must be strictly positive."
            );
        }

        if (krconfig.the_lambda < 1.0) {
            throw std::invalid_argument(
                "The lambda must be >= 1."
            );
        }

        const long steps = treeconfig.steps;
        const double dt = option.maturity()/static_cast<double>(steps);
        const double the_lambda = krconfig.the_lambda;
        const double mu = market.rate - market.dividend_yield - 0.5 * market.volatility * market.volatility;
        double discount = std::exp(-market.rate * dt);

        double u = std::exp(the_lambda * market.volatility * std::sqrt(dt));
        double m = 1.0;
        double d = 1 / u;
        double p_u = 1/(2 * the_lambda * the_lambda) + mu * std::sqrt(dt) / (2 * the_lambda * market.volatility);
        double p_d = 1/(2 * the_lambda * the_lambda) - mu * std::sqrt(dt) / (2 * the_lambda * market.volatility);
        double p_m = 1- 1/(the_lambda * the_lambda);

        if (std::fabs(p_u + p_d + p_m - 1) > 1e-12) {
            throw std::invalid_argument(
                "Kamrad-Ritchken probabilities do not sum to one.");
        }

        const TriTreeParameters treeparams{
            .u = u,
            .m = m,
            .d = d,
            .p_u = p_u,
            .p_m = p_m,
            .p_d = p_d
        };

        return trinomial_price(option, market, treeconfig, treeparams);
    }

    TrinomialResult trinomial_evaluate(const VanillaOption &option, const MarketData &market,
        const TreeConfig &treeconfig, const TriTreeParameters &treeparams) {

        if (treeparams.p_u <= 0.0 || treeparams.p_u >= 1.0) {
            throw std::invalid_argument(
                "p_u probability is outside ]0, 1[.");
        }
        if (treeparams.p_m <= 0.0 || treeparams.p_m >= 1.0) {
            throw std::invalid_argument(
                "p_m probability is outside ]0, 1[.");
        }
        if (treeparams.p_d <= 0.0 || treeparams.p_d >= 1.0) {
            throw std::invalid_argument(
                "p_d probability is outside ]0, 1[.");
        }
        TrinomialResult result;

        const long steps = treeconfig.steps;
        const double dt = option.maturity()/static_cast<double>(steps);
        const double discount = std::exp(-market.rate * dt);

        // Option value container calibrated to the number of steps + 1
        std::vector<double> option_values(
            static_cast<std::size_t>(2 *treeconfig.steps +1 ));

        // Terminal values
        double node_spot = market.spot * std::pow(treeparams.d, treeconfig.steps );

        // Terminal payoff
        double current_spot = node_spot;
        for (auto& value : option_values) {
            value = option.payoff(current_spot);
            current_spot *= treeparams.u;
        }

        // Exercise
        const bool is_american = option.exercise_style() == discrete_models::ExerciseStyle::American;

        // Backward induction
        for (int i :std::views::iota(0, treeconfig.steps) | std::views::reverse) {
            double base_spot = market.spot * std::pow(treeparams.d, i);
            double current_spot = base_spot;
            // Inner loop ; zipping index j with the required option_values which is updated
            // views::take(2*i+1) enforces that only relevant elements at step i are taken into account
            for (auto [j, value] : std::views::zip(std::views::iota(0, 2 * i + 1), option_values |
                std::views::take(2 * i + 1))) {
                const auto idx = static_cast<std::size_t>(j);
                assert(idx + 2 < option_values.size());
                const double continuation_value =
                    discount * (treeparams.p_d * option_values.at(idx)
                    + treeparams.p_m * option_values.at(idx + 1)
                    + treeparams.p_u * option_values.at(idx + 2)
                );
                const double intrinsic = option.payoff(current_spot);
                if (is_american) {
                    value = std::max(continuation_value,intrinsic);
                } else {
                    value = continuation_value;
                }
                if (i == 0) {
                    result.root_intrinsic = intrinsic;
                    result.root_continuation = continuation_value;
                }
                current_spot *= treeparams.u;
            }
        }
        result.price = option_values.front();
        return result;
    }

    double trinomial_price(const VanillaOption& option, const MarketData& market,
        const TreeConfig& config, const TriTreeParameters& TreeParams) {
        return trinomial_evaluate(
                option,
                market,
                config,
                TreeParams
            ).price;
    }

}