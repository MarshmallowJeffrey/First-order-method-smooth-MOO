"""Checkpoint logging with training-time accounting.

Training wall/CPU time excludes the time spent inside checkpoint() (metric evaluation) and the final
reward reporting in finish().  finish() writes <path>.npz (the output policies theta, their objectives F, Jacobians J and
rewards) and <path>.json (settings, checkpoints, timings, the run identity of mogym.identity and the SHA-256 of
the .npz).
"""
import json
import time

import numpy as np

from . import identity
from .metrics import reporting_metric_gram


class Recorder:
    def __init__(self, oracle, config, save_arrays=True, model=None, run_spec=None):
        self.oracle = oracle; self.config = config; self.rows = []; self.save_arrays = save_arrays
        self.model = model; self.run_spec = run_spec
        self.evalwall = 0.; self.evalcpu = 0.
        self.t0 = time.perf_counter(); self.c0 = time.process_time()

    def checkpoint(self, size, F, J, count, metric=None, kind=None):
        """Records the reported metric of the current bundle (size policies; Jacobians J): metric() if given,
        otherwise the exact K=2 value from J (SURF)."""
        trainwall = time.perf_counter() - self.t0 - self.evalwall
        traincpu = time.process_time() - self.c0 - self.evalcpu
        t, c = time.perf_counter(), time.process_time()
        gn, lam = metric() if metric is not None else reporting_metric_gram(J @ J.transpose(0, 2, 1))
        self.last_F, self.last_J = F, J
        row = dict(component_gradients=count, train_wall=trainwall, train_cpu=traincpu, gn=float(gn),
                   lambda_metric=lam.tolist(), bundle_size=size)
        if kind is not None:
            row['kind'] = kind
        self.rows.append(row)
        self.evalwall += time.perf_counter() - t; self.evalcpu += time.process_time() - c
        return gn

    def finish(self, path, theta, F, J, extra=None):
        t = time.perf_counter(); c = time.process_time()
        rr = (np.array([self.oracle.evaluate(x, False)[1] for x in theta]) if self.save_arrays
              else np.empty((0,)))
        meta = dict(config=self.config, checkpoints=self.rows, evaluation_wall=self.evalwall,
                    evaluation_cpu=self.evalcpu,
                    reward_reporting_wall=time.perf_counter() - t, reward_reporting_cpu=time.process_time() - c,
                    total_joint_calls=self.oracle.calls, total_joint_oracle_wall=self.oracle.seconds)
        if extra:
            meta.update(extra)
        path.parent.mkdir(exist_ok=True, parents=True)
        if self.run_spec is not None:
            meta['identity'] = identity.run_identity(self.run_spec, self.model)
        if self.save_arrays:
            np.savez_compressed(path.with_suffix('.npz'), theta=theta, F=F, J=J, rewards=rr)
            meta['npz_sha256'] = identity.file_sha256(path.with_suffix('.npz'))
        path.with_suffix('.json').write_text(json.dumps(meta, indent=2))
        return meta
