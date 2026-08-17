#include <pybind11/pybind11.h>
#include "binomial_pricer.h"
#include "convergence_results.h"

namespace py = pybind11;
using namespace discrete_models;

PYBIND11_MODULE(discrete_models, module) {
    module.doc() = "C++ core-engine backend for discrete-time option pricing models";

    // Linkage enum ExerciseStyle
    py::enum_<ExerciseStyle>(module, "ExerciseStyle")
        .value("European", ExerciseStyle::European)
        .value("American", ExerciseStyle::American);

    // Linkage ABC class Payoff --> no ctor linked (ABC)
    py::class_<Payoff, std::shared_ptr<Payoff>>(module, "Payoff")
        .def("__call__", &Payoff::operator()); // __call__ Python dunder method

    // Linkage class PayoffCall public Payoff
    py::class_<PayoffCall, Payoff, std::shared_ptr<PayoffCall>>(module, "PayoffCall")
        .def(py::init<double>(),py::arg("strike")); // py::arg names arguments in Python

    // Linkage class PayoffPut public Payoff
    py::class_<PayoffPut, Payoff, std::shared_ptr<PayoffPut>>(module, "PayoffPut")
        .def(py::init<double>(),py::arg("strike"));

    // Linkage class VanillaOption
    py::class_<VanillaOption>(module, "VanillaOption")
    // Passing a const ref (const Payoff&), pybind11 will temp copy the objet.
    // C++ Ctor calls then payoff.clone() to deep copy.
        .def(py::init<const Payoff&, double, ExerciseStyle>(),
        py::arg("payoff"),
        py::arg("maturity"),
        py::arg("exercise_style")
        )
        .def("payoff", &VanillaOption::payoff, py::arg("spot"))
        .def_property_readonly("maturity", &VanillaOption::maturity)  // Available like an attribute in Python
        .def_property_readonly("exercise_style", &VanillaOption::exercise_style);

    // Linkage of the MarketData structure
    py::class_<MarketData>(module, "MarketData")
        .def(py::init<>()) //Empty MarketData created !! No enough for an aggregated structure
        .def(py::init([] (const double spot,
                            const double rate,
                            const double dividend_yield,
                            const double volatility) {
                                return MarketData {
                                    spot,rate,dividend_yield,volatility
                                };
                            }
                            ),
                            py::arg("spot"),
                            py::arg("rate"),
                            py::arg("dividend_yield"),
                            py::arg("volatility")
        )
        .def_readwrite("spot", &MarketData::spot)
        .def_readwrite("rate", &MarketData::rate)
        .def_readwrite("dividend_yield", &MarketData::dividend_yield)
        .def_readwrite("volatility", &MarketData::volatility);

    // Linkage of TreeConfig
    py::class_<TreeConfig>(module, "TreeConfig")
        .def(py::init<>()) //Empty MarketData created !! No enough for an aggregated structure
        .def(py::init( [] (const int steps) {
            return TreeConfig {
                steps
            };
            }
            ),
            py::arg("steps")
        )
        .def_readwrite("steps", &TreeConfig::steps);

    // CRR
    module.def("CRR_cpp_price", &CRR_price, py::arg("option"), py::arg("market"),
        py::arg("treeconfig"));

    // JR
    module.def("JR_cpp_price", &JR_price, py::arg("option"), py::arg("market"),
        py::arg("treeconfig"));

    // TIAN93
    module.def("TIAN93_cpp_price", &TIAN93_price, py::arg("option"), py::arg("market"),
        py::arg("treeconfig"));

    // Linkage of the ConvergenceResults structure
    py::class_<ConvergenceResults>(module, "ConvergenceResults")
        .def(py::init<>()) //Empty ConvergenceResults created !! No enough for an aggregated structure
        .def(py::init([] (std::vector<int> steps,
                            std::vector<double> crr,
                            std::vector<double> crr_averaged,
                            std::vector<double> jarrow_rudd,
                            std::vector<double> tian93) {
                                return ConvergenceResults {
                                    steps,crr,crr_averaged,jarrow_rudd, tian93
                                };
                            }
                            ),
                            py::arg("steps"),
                            py::arg("crr"),
                            py::arg("crr_averaged"),
                            py::arg("jarrow_rudd"),
                            py::arg("tian93")
        )
        .def_readwrite("steps", &ConvergenceResults::steps)
        .def_readwrite("crr", &ConvergenceResults::crr)
        .def_readwrite("crr_averaged", &ConvergenceResults::crr_averaged)
        .def_readwrite("jarro_rudd", &ConvergenceResults::jarrow_rudd)
        .def_readwrite("tian93", &ConvergenceResults::tian93);

    // Convergence functions
    module.def("Binomial_Convergence", &binomial_convergence, py::arg("option"), py::arg("market"),
        py::arg("steps_grid"));

    // TEMP Convergence functions
    module.def("Binomial_Convergence_Unthreaded", &binomial_convergence_unthreaded, py::arg("option"),
        py::arg("market"),
        py::arg("steps_grid"));
}

