# discrete-time-option-pricing
This repository implements classical tree-based option pricing models, with an emphasis on clarity, inspectability, and clean separation between pricing logic, validation, and visualization.

## Repository structure

- `src/payoffs.py`: payoff functions
- `src/validation.py`: input validation
- `src/dates.py`: time-to-maturity helpers
- `src/pricers.py`: tree construction and pricing logic
- `src/visualization.py`: tree rendering utilities
- `notebooks/`: demonstrations and model comparisons
- `tests/`: unit tests

## Planned extensions

- Tian, Leiser-Reimer
- Trinomial trees
- Convergence analysis against Black-Scholes
- Continuous dividend yield
- FX-style carry conventions
- Cleaner model diagnostics
- Optional lightweight interface for demonstration

## Status

Work in progress.

## References

- Tomas Björk, *Arbitrage Theory in Continuous Time*
- Steven E. Shreve, *Stochastic Calculus for Finance I*
- Mark S. Joshi, *The Concepts and Practice of Mathematical Finance*