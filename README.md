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

After activating the environment, install the package using the following command:
```bash
git clone git+https://github.com/comp-stat/ACCORD.git accord
cd accord
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
