"""airspeed-velocity benchmark suite for mmml.

Every module here benchmarks code that lives in this repository — the PhysNet /
SpookyNet JAX models, the CHARMM-compatible MM kernels, the host neighbour-list
builders, the ``mmml.md`` driver, and the SHAKE/RATTLE constraints. Third-party
libraries are only ever exercised through an mmml entry point.

Run them with ``make bench`` or, on a GPU node, ``benchmarks/gpu_bench.py``
(see ``benchmarks/README.md``); they are deliberately *not* wired into CI.
"""
