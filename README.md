# discrete-time-option-pricing
This repository implements classical tree-based option pricing models, with an emphasis on clarity, inspectability, and clean separation between pricing logic, validation, and visualization.

## Repository structure

- `src/payoffs.py`: payoff functions
- `src/validation.py`: input validation
- `src/dates.py`: time-to-maturity helpers
- `src/pricers.py`: tree construction and pricing logic
- `src/visuals.py`: tree rendering utilities
- `notebooks/`: demonstrations and model comparisons
- `tests/`: unit tests
- `streamlit_app/` : webapp
- `figures/` : samples
- `references/` : research papers
- `examples/` : test drive

## Streamlit application
The application provides an interactive interface for CRR and Jarrow–Rudd
binomial option-pricing models.

![Application preview](figures/Appli001.png)
![Application preview](figures/Appli002.png)

python -m streamlit run app_v1.py

## Test drives
python -m pytest -q

## Planned extensions

- Tian, Leiser-Reimer
- Trinomial trees
- Convergence analysis against Black-Scholes
- Continuous dividend yield
- FX-style carry conventions
- Cleaner model diagnostics
- Optional lightweight interface for demonstration

## References

- Tomas Björk, *Arbitrage Theory in Continuous Time*
- Steven E. Shreve, *Stochastic Calculus for Finance I*
- Mark S. Joshi, *The Concepts and Practice of Mathematical Finance*
- John C. Cox, Stephen A. Ross, Mark Rubinstein, *Option Pricing : A Simplified Approch* (1979)
- Robert Jarrow, Andrew Rudd, *Approximate Option Valuation for Arbitrary Stochastic Processes* (1982)
- Dietmar Leisen, Matthias Reimer, *Binomial Models for Option Valuation - Examining and Improving Convergence* (1995)
- Yisong Tian, *A Modified Lattice Approach to Option Pricing* (1993, 1999)

## Status

Work in progress.

