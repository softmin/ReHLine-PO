# TODO

## Main Logic

1. Run benchmark against competing methods (Palomar's BSUM-MM, Potaptchik's IPM, MOSEK, CVXPY)
    - Can start with simulated data, then do it on real data later
2. Build a mini-class to solve mean-variance portfolio with piecewise linear transaction costs
    - Can start with the simplest quadratic utility maximization, and try out max-return, min-variance, max-ratio formulations
    - Then try out other optimization problems such as CVaR, CDaR, index tracking
3. Add factor analysis into the equation for potential speed-ups
4. Add holding cost

## Good development
1. Add CI-CD jazz
2. Github workflow
3. Create documentation page

## ReHLine-python optimization
1. Add practical optimizations
    - Separate variable bounds from the constraint matrix A
    - Compressed forms for A (triplets)
    - Classify different forms of constraints (equality, free, lower bounded, upper bounded) [similar to mosek API]


1. Write math report on why shifting works in `ReHLineLinear`
2. Run benchmarks CVXPY vs MOSEK vs BSUM vs ReHLine-PO on the following problems:
    - Max-return, min-risk, max-return-per-risk
    - Later on CVaR, CDaR, index tracking tools
3. Factor analysis
