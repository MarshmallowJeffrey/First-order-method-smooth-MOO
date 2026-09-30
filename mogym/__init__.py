"""Adaptive bundle method, uniform discretization and SURF on four MO-Gymnasium tasks.

  envs            finite MDP models of the tasks
  oracle          exact objectives and analytic Jacobians
  adam            Adam inner solver
  lambda_solvers  preference-weight solvers (K=2 envelope, K=3 subdivision, multistart CCP)
  metrics         reported metric max_lambda GN(lambda, B)
  plateau         run-to-plateau stopping rule and plotted point
  recorder        checkpoints and training-time accounting
  adaptive        adaptive bundle method (paper Algorithm 1)
  uniform         uniform discretization (paper Algorithm 7)
  surf            SURF Algorithm 1 (K=2)
  config          settings of the reported runs
  points          reading the runs back (plotted points, adaptive curve, budget)
"""
