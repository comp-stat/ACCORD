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
pip install git+https://github.com/comp-stat/ACCORD.git@v1.0.0
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

Output Files
-------------------
When running the program with output filename specified as `output.csv`, the following files will be generated:

- `output.csv`: Contains pairwise variable statistics such as precision values, correlations, and signs.
- `output.npy`: A NumPy binary file containing the estimated precision matrix (`omega`).
- `output_epBIC.csv`: A CSV file storing the extended pseudo-BIC(epBIC) values for different regularization parameter combinations.

### `output.npy`
- A binary NumPy file storing the estimated **omega matrix**, typically of shape `(p, p)`.
- This matrix represents the learned inverse covariance structure, encoding conditional independencies among variables.
- This file can also be used as a warm-start by supplying it via the `--warmup-file` option

### `output_epBIC.csv`
- A CSV file recording the **epBIC** (extended pseudo-BIC) values for different combinations of regularization parameters `lam1` and `lam2`.
- Each row typically corresponds to a specific pair `(lam1, lam2)` and its associated epBIC score, which is used for model selection.

### `output.csv`
This file contains estimated relationships between variable pairs. Each row corresponds to a pair of variables, and each column is defined as follows:

- **V1**: Identifier (name or index) of the first variable in the pair.
- **V2**: Identifier (name or index) of the second variable in the pair.
- **Precision.value**: Off-diagonal entry in the estimated precision (inverse covariance) matrix corresponding to (V1, V2). Indicates conditional dependence strength.
- **Partial.Corr**: Partial correlation coefficient between V1 and V2, derived from the precision matrix. 
- **Pearson.Corr**: Standard Pearson correlation between V1 and V2, derived from the raw data.
> **Note:** When the `--sparse` option is enabled, the output is restricted to variable pairs with nonzero entries in the estimated precision matrix (`Precision.value`). As a result, the `Pearson.Corr`, `AbsPearsonCorr`, and `SignPearsonCorr` columns may be partially missing.
- **AbsPartialCorr**: Absolute value of the partial correlation (i.e., |Partial.Corr|). Used to evaluate the magnitude of conditional dependence.
- **SignPartialCorr**: Sign of the partial correlation coefficient (+1, -1), indicating the direction of the conditional relationship.
- **AbsPearsonCorr**: Absolute value of the Pearson correlation (i.e., |Pearson.Corr|). Used to assess the strength of marginal association.
- **SignPearsonCorr**: Sign of the Pearson correlation coefficient (+1, -1), indicating the direction of the marginal association.

Directory Structure
-------------------

- __src__: It contains C++ source code for various versions of ACCORD algorithms.
- [__gaccord__](./gaccord/README.md): It contains Python classes for the ACCORD algorithms.
