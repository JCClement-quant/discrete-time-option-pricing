#pragma once
#include <binomial_pricer.h>
#include <cmath>

namespace discrete_models {
    struct TrinomialResult {
        double price;
        double root_continuation;
        double root_intrinsic;
    };

    struct TriTreeParameters {
        double u;
        double m;
        double d;
        double p_u;
        double p_m;
        double p_d;
    };

    struct KRconfig {
        double the_lambda = std::sqrt(3.0/2.0);
    };

    [[nodiscard]]
    double Kamrad_Ritchken(const VanillaOption& option, const MarketData& market,
        const TreeConfig& config, const KRconfig& krconfig);

    [[nodiscard]]
    double trinomial_price(const VanillaOption& option, const MarketData& market,
        const TreeConfig& config, const TriTreeParameters& TreeParams);
}