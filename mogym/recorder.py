"""Checkpoint logging with training-time accounting.

Training wall/CPU time excludes the time spent inside checkpoint() (metric evaluation) and the final
reward reporting in finish(); oracle calls made for evaluation are counted separately and never enter
Gradient Calls.  finish() writes <path>.json (settings, checkpoints, timings) and <path>.npz (the output
policies theta, their objectives F, Jacobians J and rewards).
"""
import json
import time

import numpy as np

from .lambda_solvers import evaluate_gram


class Recorder:
    def __init__(self, oracle, config, save_arrays=True):
        self.oracle = oracle; self.config = config; self.rows = []; self.save_arrays = save_arrays
        self.evalwall = 0.; self.evalcpu = 0.; self.evalcalls = 0
        self.t0 = time.perf_counter(); self.c0 = time.process_time()

    def checkpoint(self, theta, F=None, J=None, count=0, force_exact=None):
        trainwall = time.perf_counter() - self.t0 - self.evalwall
        traincpu = time.process_time() - self.c0 - self.evalcpu
        t, c = time.perf_counter(), time.process_time(); before = self.oracle.calls
        theta = np.atleast_2d(theta)
        if J is None:
            fj = [self.oracle(x) for x in theta]
            F = np.array([z[0] for z in fj]); J = np.array([z[1] for z in fj])
        if force_exact is None:
            gn, lam, upper = evaluate_gram(J @ J.transpose(0, 2, 1))
        else:
            gn, lam, upper = force_exact()
        self.last_F, self.last_J = F, J
        self.rows.append(dict(component_gradients=count, train_wall=trainwall, train_cpu=traincpu,
                              gn=float(gn), gn_upper=float(upper), lambda_metric=lam.tolist(),
                              bundle_size=len(theta)))
        self.evalcalls += self.oracle.calls - before
        self.evalwall += time.perf_counter() - t; self.evalcpu += time.process_time() - c
        return gn

    def finish(self, path, theta, F, J, extra=None):
        t = time.perf_counter(); c = time.process_time()
        rr = (np.array([self.oracle.evaluate(x, False)[1] for x in theta]) if self.save_arrays
              else np.empty((0,)))
        meta = dict(config=self.config, checkpoints=self.rows, evaluation_wall=self.evalwall,
                    evaluation_cpu=self.evalcpu, evaluation_joint_calls=self.evalcalls,
                    reward_reporting_wall=time.perf_counter() - t, reward_reporting_cpu=time.process_time() - c,
                    total_joint_calls=self.oracle.calls, total_joint_oracle_wall=self.oracle.seconds)
        if extra:
            meta.update(extra)
        path.parent.mkdir(exist_ok=True, parents=True)
        if self.save_arrays:
            np.savez_compressed(path.with_suffix('.npz'), theta=theta, F=F, J=J, rewards=rr)
        path.with_suffix('.json').write_text(json.dumps(meta, indent=2))
        return meta
