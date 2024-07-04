# TODO

## ReHLine-PO extensions
1. Build a mini-class to solve mean-variance portfolio with piecewise linear transaction costs
    - Can start with the simplest quadratic utility maximization, and try out max-return, min-variance, max-ratio formulations
    - Then try out other optimization problems such as CVaR, CDaR, index tracking
2. Add factor analysis into the equation for potential speed-ups
3. Add holding cost

## Good development
1. Add CI-CD jazz
2. Github workflow

## ReHLine-python optimization
1. Add practical optimizations
    - Separate box constraints from the matrix A to optimize ReHLine
    - Allow compressed forms for A (triplets)
    - Classify different forms of constraints (equality, free, lower bounded, upper bounded) [similar to mosek API]
