#include <cmath>

double static peizer_pratt_inversion(double z, int N) {
    if (std::abs(z) <1e-15) return 0.5;

    auto d = N + (1.0 / 3.0) + (0.1 / (N + 1.0));
    auto term = z * z / (d * d) * (N + (1.0 / 6.0) );
    auto sign = -1;
    if (z > 0) sign = 1;

    return 0.5 + sign * 0.5 * std::sqrt(1 - std::exp(-term));
}