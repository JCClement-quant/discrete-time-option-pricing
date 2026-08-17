import numpy as np

def peizer_pratt_inversion(z: float, n: int) -> float:
    """Peizer-Pratt nnversion (PP2) for visual trees."""
    if abs(z) < 1e-15:
        return 0.5

    denominator = n + (1.0 / 3.0) + (0.1 / (n + 1.0))
    term = (z / denominator) ** 2 * (n + (1.0 / 6.0))
    sign = 1.0 if z > 0 else -1.0

    return 0.5 + sign * 0.5 * np.sqrt(1.0 - np.exp(-term))