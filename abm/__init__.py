"""Adaptive bundle method (ABM) for smooth multi-objective optimization: the MNIST experiments.

Modules
-------
config      settings of the paper's experiments (digits, rho, budgets, seeds, schedules)
data        MNIST loading
model       the locally connected network and its initial point
objective   the pooled objectives F_k = L_k + rho * L_pool + (mu/2) ||theta||^2 and their oracles
steppers    step rules (constant step, Barzilai-Borwein, AdaGrad, Adam)
training    one SVRG segment, the budget meter, run records
ccp         the multistart CCP lambda-search of the adaptive method
methods     the adaptive bundle method, uniform discretization and SURF
meter       worst-case gradient norm of a bundle: exact for K = 2, a lower bound for K = 3
analysis    plateau test, markers and trend fits
fronts      linear scalarization fronts
screening   conflict screening of digit pairs and triples
labels      label placement for the trend figures
"""
