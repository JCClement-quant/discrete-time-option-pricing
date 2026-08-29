from enum import Enum
from _collections_abc import Sequence
from typing import Any, overload
import numpy as np
import numpy.typing as npt

# Helps PyCharm to acknowledge module discrete_models

class ExerciseStyle(Enum):
    European: ExerciseStyle
    American: ExerciseStyle

class Payoff:
    def __call__(self, spot: float)-> float:...

class PayoffCall(Payoff):
    def __init__(self, strike: float) -> None:...

class PayoffPut(Payoff):
    def __init__(self, strike: float) -> None:...

class VanillaOption:
    def __init__(self, payoff: Payoff, maturity : float, exercise_style : ExerciseStyle) -> None:...
    def payoff(selfself, spot: float) -> float:...
    @property
    def strike(self) -> float: ...
    @property
    def exercise_style(self) -> ExerciseStyle:...
    @property
    def maturity(self) -> float:...

class MarketData:
    spot: float
    rate : float
    dividend_yield :float
    volatility : float
    def __init__(self)-> None:...

class TreeConfig:
    steps: int
    def __init__(self)->None:...

def CRR_cpp_price(
    option: VanillaOption,
    market: MarketData,
    config: TreeConfig,
) -> float: ...

def JR_cpp_price(
    option: VanillaOption,
    market: MarketData,
    config: TreeConfig,
) -> float: ...

def TIAN93_cpp_price(
    option: VanillaOption,
    market: MarketData,
    config: TreeConfig,
) -> float: ...

def LR_cpp_price(
    option: VanillaOption,
    market: MarketData,
    config: TreeConfig,
) -> float: ...

def LR_root(
    option: VanillaOption,
    market: MarketData,
    config: TreeConfig,
) -> float: ...


StepsGrid = Sequence[int] | npt.NDArray[np.integer[Any]]

class ConvergenceResults:
    steps: list[int]
    crr: list[float]
    crr_averaged: list[float]
    jarrow_rudd: list[float]
    tian93: list[float]
    leisen_reimer:list[float]

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(
        self,
        steps: StepsGrid,
        crr: Sequence[float],
        crr_averaged: Sequence[float],
        jarrow_rudd: Sequence[float],
        tian93: Sequence[float],
        leisen_reimer: Sequence[float],
    ) -> None: ...

def Binomial_Convergence(
    option: VanillaOption,
    market: MarketData,
    steps_grid: StepsGrid,
) -> ConvergenceResults: ...


def Binomial_Convergence_Unthreaded(
    option: VanillaOption,
    market: MarketData,
    steps_grid: StepsGrid,
) -> ConvergenceResults: ...

# ===================================================== #
#                       NOTES                           #
# ===================================================== #
#                                                       #
# Inheritance :                                         #
#   class PayoffCall(Payoff) brings __call__ from       #
#   class Payoff to classes PayOffCall and PayOffPut    #
# Structures MarketData and TreeConfig :                #
#   variables declared with their types                 #
# Default ctor :                                        #
#   for MarketData is mandatory even if MarketData() is #
#   void in C++... PyCharm needs the line               #
#   def __init__(self) -> None: ... to allow ()         #
#   instantiation in Python.                            #
# Structure ConvergenceResults                          #
#   std::vector is genuinely converted into a native    #
#   Python container, list. However, the return is      #
#   Sequence[int] | npt.NDArray[np.integer[Any]] to     #
#   widen the range of choices of Python collectors on  #
#   the user side : list, tuple, Numpy.                 #
# ===================================================== #