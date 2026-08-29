#include "binomial_pricer.h"
#include "convergence_results.h"
#include <iostream>
#include <chrono>

using std::cout;
using std::endl;

int main() {

    const discrete_models::PayoffCall payoff_call(100.00);
    const discrete_models::PayoffPut payoff_put(100.00);

    const discrete_models::VanillaOption option_call_eu{payoff_call, 1.0,
        discrete_models::ExerciseStyle::European};
    const discrete_models::VanillaOption option_put_us{payoff_put, 1.0,
        discrete_models::ExerciseStyle::American};
    const discrete_models::VanillaOption option_call_us{payoff_call, 1.0,
        discrete_models::ExerciseStyle::American};
    const discrete_models::VanillaOption option_put_eu{
        payoff_put, 1.0,
        discrete_models::ExerciseStyle::European};

    const discrete_models::MarketData market_data{
        .spot = 120.00,
        .rate = 0.05,
        .dividend_yield = 0.50,
        .volatility = 0.20
    };

    const discrete_models::TreeConfig tree_config{
        .steps = 10001
    };

    /*
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
    price = CRR_price(option_call_us, market_data, tree_config);
    cout << "Prix CRR du call americain : " << price << endl;
    price = JR_price(option_call_us, market_data, tree_config);
    cout << "Prix JR du call americain : " << price << endl;
    price = TIAN93_price(option_call_us, market_data, tree_config);
    cout << "Prix TIAN93 du call americain : " << price << endl;
    price = CRR_price(option_put_eu, market_data, tree_config);
    cout << "Prix CRR du put europeen : " << price << endl;
    price = JR_price(option_put_eu, market_data, tree_config);
    cout << "Prix JR du put europeen : " << price << endl;
    price = TIAN93_price(option_put_eu, market_data, tree_config);
    cout << "Prix TIAN93 du put europeen : " << price << endl;


    std::vector<int> steps = {5,10,15,25,50,100,250,500,1000,2500,5000,10000,25000,50000};
    auto start_thread = std::chrono::high_resolution_clock::now();
    ConvergenceResults result = binomial_convergence(option_call_eu,market_data, steps);

    for (std::size_t i = 0; i < steps.size(); ++i) {
        cout << steps[i];
        cout << " CRR " << result.crr[i];
        cout << " CRR AVG " << result.crr_averaged[i];
        cout << " Jarrow-Rudd " << result.jarrow_rudd[i];
        cout << " Tian93 " << result.tian93[i] << endl;
    }
    auto end_thread = std::chrono::high_resolution_clock::now();
    auto time_threaded = std::chrono::duration_cast<std::chrono::nanoseconds>
        (end_thread - start_thread).count();
    cout << endl << "********** " << time_threaded<< " **********"  << endl << endl;

    auto start_unthread = std::chrono::high_resolution_clock::now();
    ConvergenceResults result_unthread = binomial_convergence_unthreaded(option_call_eu,market_data, steps);

    for (std::size_t i = 0; i < steps.size(); ++i) {
        cout << steps[i];
        cout << " CRR " << result_unthread.crr[i];
        cout << " CRR AVG " << result_unthread.crr_averaged[i];
        cout << " Jarrow-Rudd " << result_unthread.jarrow_rudd[i];
        cout << " Tian93 " << result_unthread.tian93[i] << endl;
    }
    auto end_unthread = std::chrono::high_resolution_clock::now();
    auto time_unthreaded = std::chrono::duration_cast<std::chrono::nanoseconds>
    (end_unthread - start_unthread).count();
    cout << endl << "********** " << time_unthreaded<< " **********"  << endl << endl;
    const double speedup = static_cast<double>(time_unthreaded)/ static_cast<double>(time_threaded);

    const double efficiency = speedup / 4.0;
    cout << "speedup = " << speedup << endl << endl;
    cout << "efficency = " << efficiency << endl << endl;
    */

    double price = CRR_price(option_call_eu, market_data, tree_config);
    cout << "Prix CRR du call europeen : " << price << endl;
    price = LR_price(option_call_eu, market_data, tree_config);
    cout << "Prix LR du call europeen : " << price << endl;
    price = CRR_price(option_call_us, market_data, tree_config);
    cout << "Prix CRR du call americain : " << price << endl;
    price = LR_price(option_call_us, market_data, tree_config);
    cout << "Prix LR du call americain : " << price << endl;
    price = CRR_price(option_put_eu, market_data, tree_config);
    cout << "Prix CRR du put europeen : " << price << endl;
    price = LR_price(option_put_eu, market_data, tree_config);
    cout << "Prix LR du put europeen : " << price << endl;
    price = CRR_price(option_put_us, market_data, tree_config);
    cout << "Prix CRR du put americain : " << price << endl;
    price = LR_price(option_put_us, market_data, tree_config);
    cout << "Prix LR du put americain: " << price << endl;
    cout << LR_root_exercise_gap(option_call_us, market_data, tree_config) << endl;
    cout << LR_root_exercise_gap(option_put_us, market_data, tree_config) << endl;
    return 0;
}
