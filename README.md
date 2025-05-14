ACCORD Software
===============

![Build](https://github.com/gardenk11181/ACCORD/actions/workflows/python-package.yml/badge.svg)
![Format](https://github.com/gardenk11181/ACCORD/actions/workflows/auto-format.yml/badge.svg)

This repository provides a command-line interface and C++ backend for ACCORD algorithm from the following paper.

📄 **Learning Massive-scale Partial Correlation Networks in Clinical Multi-omics Studies with HP-ACCORD**  
*Sungdong Lee, Joshua Bang, Youngrae Kim, Hyungwon Choi, Sang-Yun Oh, Joong-Ho Won*
arXiv: [2412.11554](https://arxiv.org/abs/2412.11554)

Installation
---------------

We recommend using a Python virtual environment such as venv.
```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activating the environment, install the package using the following command:
```bash
pip install git+https://github.com/comp-stat/ACCORD.git@v1.0.0-alpha
```

How to Run
---------------

Get options using the following command:
```bash
accord --help
```

Example command:
```bash
accord --input-file input.csv --output-file output.csv
```

Directory Structure
-------------------

- __src__: It contains C++ source code for various versions of ACCORD algorithms.
- [__gaccord__](./gaccord/README.md): It contains Python classes for the ACCORD algorithms.
