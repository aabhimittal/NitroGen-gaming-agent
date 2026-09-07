"""Deployment hardening: everything between a trained checkpoint and a real game.

The research path (``model.act(frame)``) assumes clean pixels, a sampler that
always answers, and an environment that accepts any vector in ``[-1, 1]^18``.
None of that holds against a real title captured off a real screen. This package
supplies the missing layer:

    :mod:`~nitrogen.deploy.frames`      capture-pipeline hardening (dtype, layout,
                                        NaN, resolution, black/frozen detection)
    :mod:`~nitrogen.deploy.safety`      hardware-legal, human-plausible actions
    :mod:`~nitrogen.deploy.controller`  chunk -> per-tick execution, ensembling,
                                        latency compensation, starvation policy
    :mod:`~nitrogen.deploy.runtime`     the supervised loop, with a circuit
                                        breaker and latency metrics

Only :mod:`~nitrogen.deploy.runtime`'s :meth:`PolicyRuntime.from_model` needs
torch; the rest is numpy, so the hardening layer can be tested (and reused in a
capture process) without loading a model.

Typical usage::

    runtime = PolicyRuntime.from_model(model, seed=0)
    while not done:
        report = runtime.step(capture.grab())     # never raises, never stalls
        frame, done, info = env.step(report.action)
    print(runtime.metrics.summary())
"""

from nitrogen.deploy.controller import ChunkExecutor
from nitrogen.deploy.frames import FrameHealth, FrameSanitizer, letterbox, resize_nearest
from nitrogen.deploy.runtime import PolicyRuntime, RuntimeMetrics, StepReport
from nitrogen.deploy.safety import ActionGuard, GamepadLimits

__all__ = [
    "ChunkExecutor",
    "FrameHealth",
    "FrameSanitizer",
    "letterbox",
    "resize_nearest",
    "PolicyRuntime",
    "RuntimeMetrics",
    "StepReport",
    "ActionGuard",
    "GamepadLimits",
]
