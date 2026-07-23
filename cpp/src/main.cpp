#include "binomial_pricer.h"
#include <iostream>

using std::cout;
using std::endl;

int main() {

    const discrete_models::PayoffCall payoff_call(100.00);
    const discrete_models::PayoffPut payoff_put(100.00);

    const discrete_models::VanillaOption option_call_eu{payoff_call, 1.0,
        discrete_models::ExerciseStyle::European};
    const discrete_models::VanillaOption option_put_us{payoff_put, 1.0,
    discrete_models::ExerciseStyle::American};

    const discrete_models::MarketData market_data{
        .spot = 100.00,
        .rate = 0.05,
        .dividend_yield = 0.0,
        .volatility = 0.20
    };

    const discrete_models::TreeConfig tree_config{
        .steps = 10000
    };

    double price = CRR_price(option_call_eu, market_data, tree_config);
    cout << "Prix CRR du call europeen : " << price << endl;
    price = JR_price(option_call_eu, market_data, tree_config);
    cout << "Prix JR du call europeen : " << price << endl;
    price = TIAN93_price(option_call_eu, market_data, tree_config);
    cout << "Prix TIAN93 du call europeen : " << price << endl;
    price = CRR_price(option_put_us, market_data, tree_config);
    cout << "Prix CRR du put americain : " << price << endl;
    price = JR_price(option_put_us, market_data, tree_config);
    cout << "Prix JR du put americain : " << price << endl;
    price = TIAN93_price(option_put_us, market_data, tree_config);
    cout << "Prix TIAN93 du put americain : " << price << endl;

    return 0;
}
