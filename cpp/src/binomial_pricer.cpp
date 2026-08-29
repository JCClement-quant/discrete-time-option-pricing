#include "binomial_pricer.h"
#include "functions.cpp"
#include <algorithm>
#include <utility>
#include <cmath>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <vector>
#include <ranges>


namespace{
    discrete_models::TreeParameters LR_Parameters(const discrete_models::VanillaOption& option,
        const discrete_models::MarketData& market,const discrete_models::TreeConfig& treeconfig) {
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
        if (treeconfig.steps % 2 == 0) {
            throw std::invalid_argument(
                "Leiser-Reimer requires an odd number of steps."
            );
        }

        auto sigma_sqrt_T = market.volatility * std::sqrt(option.maturity());
        auto d1 = (std::log(market.spot / option.strike()) + (market.rate - market.dividend_yield + 0.5 *
            market.volatility * market.volatility ) * option.maturity()) / sigma_sqrt_T;
        auto d2 = d1 - sigma_sqrt_T;

        auto p_prime = peizer_pratt_inversion(d1,treeconfig.steps);
        auto p = peizer_pratt_inversion(d2, treeconfig.steps);

        const long steps = treeconfig.steps;
        const double dt = option.maturity()/static_cast<double>(steps);
        const double carry_rate = market.rate - market.dividend_yield;
        const double a = std::exp(carry_rate * dt);
        const double u = a * p_prime / p;
        const double d = (a - p * u) / (1 - p);

        const discrete_models::TreeParameters treeparams{
            .u = u,
            .d = d,
            .p = p
        };

        if (p <= 0.0 || p >= 1.0) {
            throw std::invalid_argument(
                "LR risk-neutral probability is outside ]0, 1[.");
        }
        return treeparams;
    }
}

namespace discrete_models {
    PayoffCall::PayoffCall(double strike) : strike_(strike) {}

    double PayoffCall::operator()(const double spot) const {
        return std::max(spot - strike_, 0.0);
    }
    std::unique_ptr<Payoff> PayoffCall::clone() const {
        return std::make_unique<PayoffCall>(*this);
    }

    PayoffPut::PayoffPut(double strike) : strike_(strike) {}

    double PayoffPut::operator()(const double spot) const {
        return std::max(strike_ - spot, 0.0);
    }
    std::unique_ptr<Payoff> PayoffPut::clone() const {
        return std::make_unique<PayoffPut>(*this);
    }

    VanillaOption::VanillaOption(const Payoff &payoff, double maturity,
                                 ExerciseStyle exercise_style) : payoff_(payoff.clone()), maturity_(maturity),
                                                                       exercise_style_(exercise_style) {
    }

    VanillaOption::VanillaOption(const VanillaOption &option) : payoff_(option.payoff_->clone()),
                                                                maturity_(option.maturity_),
                                                                exercise_style_(option.exercise_style_) {
    }

    VanillaOption &VanillaOption::operator=(const VanillaOption &option) {
        if (this != &option) {
            VanillaOption copy(option);
            swap(*this, copy);
        }
        return *this;
    }

    double VanillaOption::strike() const noexcept {
        return payoff_->strike();
    }

    double VanillaOption::payoff(const double spot) const {
        return (*payoff_)(spot);
    }

    double VanillaOption::maturity() const noexcept {
        return maturity_;
    }

    ExerciseStyle VanillaOption::exercise_style() const noexcept {
        return exercise_style_;
    }

    double CRR_price(const VanillaOption &option, const MarketData &market, const TreeConfig &treeconfig) {

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

        const long steps = treeconfig.steps;
        const double dt = option.maturity()/static_cast<double>(steps);
        const double u = std::exp(market.volatility * std::sqrt(dt));
        const double d = 1 / u;
        const double carry_rate = market.rate - market.dividend_yield;
        const double growth = std::exp(carry_rate * dt);
        const double p = (growth - d) / (u - d);

        const TreeParameters treeparams{
            .u = u,
            .d = d,
            .p = p
        };

        if (p <= 0.0 || p >= 1.0) {
            throw std::invalid_argument(
                "CRR risk-neutral probability is outside ]0, 1[.");
        }

        return binomial_price(option,market,treeconfig, treeparams);
    }

    double JR_price(const VanillaOption &option, const MarketData &market, const TreeConfig &treeconfig) {

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

        const long steps = treeconfig.steps;
        const double dt = option.maturity()/static_cast<double>(steps);
        const double carry_rate = market.rate - market.dividend_yield;
        const double u = std::exp((carry_rate - 0.5 * std::pow(market.volatility,2)) * dt +
            market.volatility * std::sqrt(dt));
        const double d = std::exp((carry_rate - 0.5 * std::pow(market.volatility,2)) * dt -
            market.volatility * std::sqrt(dt));

        const double p = 0.5;

        const TreeParameters treeparams{
            .u = u,
            .d = d,
            .p = p
        };

        if (p <= 0.0 || p >= 1.0) {
            throw std::invalid_argument(
                "JR risk-neutral probability is outside ]0, 1[.");
        }

        return binomial_price(option,market,treeconfig, treeparams);
    }

    double TIAN93_price(const VanillaOption &option, const MarketData &market, const TreeConfig &treeconfig) {

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

        const long steps = treeconfig.steps;
        const double dt = option.maturity()/static_cast<double>(steps);
        const double carry_rate = market.rate - market.dividend_yield;
        const double h = std::exp(std::pow(market.volatility,2) * dt);
        const double u = 0.5 * std::exp(carry_rate * dt ) * h * (h + 1 +
            std::sqrt(std::pow(h,2) + 2 * h - 3));
        const double d = 0.5 * std::exp(carry_rate * dt ) * h * (h + 1 -
            std::sqrt(std::pow(h,2) + 2 * h - 3));

        const double p = (std::exp(carry_rate * dt) - d)/(u - d);

        const TreeParameters treeparams{
            .u = u,
            .d = d,
            .p = p
        };

        if (p <= 0.0 || p >= 1.0) {
            throw std::invalid_argument(
                "TIAN93 risk-neutral probability is outside ]0, 1[.");
        }

        return binomial_price(option,market,treeconfig, treeparams);
    }

    double LR_price(const VanillaOption &option, const MarketData &market, const TreeConfig &treeconfig) {
        const TreeParameters treeparams = LR_Parameters(option, market, treeconfig);

        if (treeparams.p <= 0.0 || treeparams.p >= 1.0) {
            throw std::invalid_argument(
                "LR risk-neutral probability is outside ]0, 1[.");
        }

        return binomial_price(option,market,treeconfig,treeparams);

    }

    BinomialResult binomial_evaluate(const VanillaOption &option, const MarketData &market, const TreeConfig &treeconfig,
        const TreeParameters &treeparams) {
        if (treeparams.p <= 0.0 || treeparams.p >= 1.0) {
            throw std::invalid_argument(
                "Risk-neutral probability is outside ]0, 1[.");
        }
        BinomialResult result;

        const long steps = treeconfig.steps;
        const double dt = option.maturity()/static_cast<double>(steps);
        const double discount = std::exp(-market.rate * dt);
        const double q = 1- treeparams.p ;

        // Option value container calibrated to the number of steps + 1
        std::vector<double> option_values(
            static_cast<std::size_t>(treeconfig.steps +1 ));

        // Terminal values
        double node_spot = market.spot * std::pow(treeparams.d, treeconfig.steps );
        const double u_over_d = treeparams.u / treeparams.d;

        // Terminal payoff
        /*
        for (auto [i, value] : std::views::zip(std::views::iota(0, treeconfig.steps + 1),
            option_values)) {
            double current_spot = node_spot * std::pow(u_over_d, i);
            value = option.payoff(current_spot);
        }
        */

        // Terminal payoff
        double current_spot = node_spot;
        for (auto& value : option_values) {
            value = option.payoff(current_spot);
            current_spot *= u_over_d;
        }

        // Exercise
        const bool is_american = option.exercise_style() == ExerciseStyle::American;
        // std::size_t early_exercise_count = 0;

        // Lowest spot at time (steps - 1)
        // double lowest_spot = market.spot * std::pow(treeparams.d, treeconfig.steps - 1);

        // Backward induction
        for ( int i :std::views::iota(0, treeconfig.steps) | std::views::reverse) {
            // S_{i,0} = S_0 * d^i
            double base_spot = market.spot * std::pow(treeparams.d, i);
            double current_spot = base_spot;
            // Inner loop ; zipping index j with the required option_values which is updated
            // views::take(i+1) enforces that only relevant elements at step i are taken into account
            for (auto [j, value] : std::views::zip(std::views::iota(0, i + 1), option_values |
                std::views::take(i+1))) {
                const double continuation_value =
                    discount * (treeparams.p * option_values[static_cast<std::size_t>(j) +1 ] +q * value);
                const double intrinsic = option.payoff(current_spot);
                if (is_american) {

                    value = std::max(continuation_value,intrinsic);
                    current_spot *= u_over_d;
                } else {
                    value = continuation_value;
                }
                if (i == 0) {
                    result.root_intrinsic = intrinsic;
                    result.root_continuation = continuation_value;
                }
            }
        }
        result.price = option_values.front();
        return result;
    }

    double LR_root_exercise_gap(const VanillaOption& option, const MarketData& market, const TreeConfig& treeconfig) {
        const TreeParameters treeparams = LR_Parameters(option, market, treeconfig);
        const auto result = binomial_evaluate(option,market,treeconfig, treeparams);
        return (result.root_continuation - result.root_intrinsic);
    }

    double binomial_price(const VanillaOption &option, const MarketData &market, const TreeConfig &treeconfig,
    const TreeParameters &treeparams) {
        return binomial_evaluate(
            option,
            market,
            treeconfig,
            treeparams
        ).price;
    }
}

