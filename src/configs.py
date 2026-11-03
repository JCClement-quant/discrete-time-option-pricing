from dataclasses import dataclass
import math

@dataclass(frozen=True)
class KRConfig:
    the_lambda: float = math.sqrt(3.0 / 2.0)

    def __post_init__(self):
        if self.the_lambda < 1.0:
            raise ValueError("KR lambda must be >= 1.")