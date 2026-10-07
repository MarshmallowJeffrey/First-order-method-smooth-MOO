"""GRAB, uniform discretization and SURF on two MO-Gymnasium tasks (FishWood, Fruit Tree).

  envs            finite MDP models of the tasks
  oracle          exact objectives and analytic Jacobians
  adam            Adam inner solver
  lambda_solvers  preference-weight solvers (K=2 envelope, multistart CCP)
  metrics         max_lambda GN(lambda, B): exact for K=2, a lower bound (pool + CCP polishing) for K>2
  bounds          K>2: upper bounds on GRAB's max_lambda GN(lambda, B_t) at every checkpoint (branch and bound)
  plateau         run-to-plateau stopping rule and plotted point
  recorder        checkpoints and training-time accounting
  adaptive        GRAB (Algorithm 1)
  uniform         uniform discretization (Algorithm 6)
  surf            SURF Algorithm 1 (K=2)
  config          settings of the reported runs
  identity        identity of a stored run (settings, source, versions, model, arrays)
  points          reading the runs back (plotted points, GRAB curve and its upper bounds)
"""
