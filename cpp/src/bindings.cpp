#include <pybind11/pybind11.h>
#include "binomial_pricer.h"

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
        .def(py::init<>()) //Empty MarketData created
        .def_readwrite("spot", &MarketData::spot)
        .def_readwrite("rate", &MarketData::rate)
        .def_readwrite("dividend_yield", &MarketData::dividend_yield)
        .def_readwrite("volatility", &MarketData::volatility);

    // Linkage of TreeConfig
    py::class_<TreeConfig>(module, "TreeConfig")
        .def(py::init<>())
        .def_readwrite("steps", &TreeConfig::steps);

    // CRR
    module.def("CRR_cpp_price", &CRR_price, py::arg("option"), py::arg("market"),
        py::arg("treeconfig"));
}

