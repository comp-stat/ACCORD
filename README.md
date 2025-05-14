ACCORD Software
===============

![Build](https://github.com/gardenk11181/ACCORD/actions/workflows/python-package.yml/badge.svg)
![Format](https://github.com/gardenk11181/ACCORD/actions/workflows/auto-format.yml/badge.svg)

This repository includes code to deploy ACCORD software.

Installation
---------------

We recommend using a Python virtual environment such as venv.

After activating the environment, install the package in development mode using the following command:
```bash
pip install -e .
```

How to Run
---------------

Get options using the following command:
```bash
python -m gaccord.cli --help
```

Example:
```bash
python -m gaccord.cli --input-file data/example.csv --output-file out.csv
```

Directory Structure
-------------------

- __src__: It contains C++ source code for various versions of ACCORD algorithms.
- [__gaccord__](./gaccord/README.md): It contains Python classes for the ACCORD algorithms.
