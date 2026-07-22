#include "binomial_pricer.h"
#include <iostream>

using std::cout;
using std::endl;

int main() {

    const discrete_models::PayoffCall payoff_call(100.00);

    const discrete_models::VanillaOption option{payoff_call, 1.0, discrete_models::ExerciseStyle::European};

    const discrete_models::MarketData market_data{
        .spot = 100.00,
        .rate = 0.05,
        .dividend_yield = 0.0,
        .volatility = 0.20
    };

    const discrete_models::TreeConfig tree_config{
        .steps = 10000
    };

    const double price = CRR_price(option, market_data, tree_config);
    cout << price << endl;

    return 0;
}
