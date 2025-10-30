User Guide
==========

This page details how to use mmml.

Overview
--------

High-level concepts, capabilities, and typical workflows with mmml.

Command-line Interface
----------------------

How to run common tasks from the CLI (training, ESP generation, conversions).

Saving Results from PySCF GPU
-----------------------------

Use the CLI flag ``--save_option`` to control how results are persisted when running
``mmml.pyscf4gpuInterface.calcs``. Supported values are ``pkl``, ``npz``, and ``hdf5``.
See :ref:`PySCF GPU Interface API <pyscf4gpu_api>` for details and examples.

Python API
----------

Programmatic usage patterns and key modules to import.

Configuration
-------------

How to configure runs, set seeds, and control resources.

Troubleshooting
---------------

Common issues and tips (imports, GPU setup, missing optional dependencies).
