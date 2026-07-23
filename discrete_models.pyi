from enum import Enum

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
#                                                       #
# ===================================================== #