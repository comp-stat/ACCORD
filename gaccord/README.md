ACCORD Software Python Classes
==============================

## 📚 Table of Contents

* [cli.py](#clipy)
* [correlation.py](#correlationpy)
* [epbic.py](#epbicpy)
* [runner.py](#runnerpy)

## cli.py

This is a command-line interface for running the Graphical ACCORD algorithm for estimating sparse inverse covariance matrices with structured penalties.

## correlation.py

This module provides utility functions to compute **partial correlation** and **simple correlation** matrices from a given precision matrix (inverse covariance matrix), commonly used in graphical models and statistical inference.

### 🧠 Functions

**`compute_partial_correlation(theta: np.ndarray) -> np.ndarray`**

Computes the **partial correlation matrix** from a given precision matrix `theta`.

* **Input**:
  `theta` — A symmetric, positive-definite **precision matrix** (inverse covariance)

* **Formula**:
  $P = - D^{-1/2}  \Theta  D^{-1/2}$
  where $D$ is a diagonal matrix of $\Theta$'s diagonal elements.


**`compute_simple_correlation(theta: np.ndarray) -> np.ndarray`**

Computes the **simple correlation matrix** from a given precision matrix by first inverting it to get the covariance matrix.

* **Input**:
  `theta` — A symmetric, positive-definite **precision matrix**

* **Process**:

  1. Invert $\Theta$ to get the covariance matrix $\Sigma$
  2. Standardize it to obtain the correlation matrix:
     $R = D^{-1/2}  \Sigma  D^{-1/2}$

## epbic.py

This Python module provides functions for computing the **extended partial Bayesian Information Criterion (epBIC)**, often used in model selection for graphical models. It includes utilities for counting non-zero elements and calculating trace-based statistics.

### 🧠 Functions

**`count_nonzero_entries(matrix: np.ndarray) -> int`**

Counts the number of **non-zero elements** in the given matrix.

- **Input**:  
  `matrix`: A NumPy array of any shape.

- **Returns**:  
  Integer count of non-zero entries in `matrix`.


**`compute_g(Xn: np.ndarray, S: np.ndarray) -> float`**

Computes the trace-based value:

\$g(\hat{\Omega})=\frac{1}{2}\text{tr}(\hat{\Omega}^T \hat{\Omega} S)$

- **Inputs**:  
  - `Xn`: Estimate of the precision matrix or transformation.  
  - `S`: Empirical covariance matrix.

- **Returns**:  
  A scalar value representing the \( g(\hat{\Omega}) \) term.


**`compute_epBIC(Xn: np.ndarray, S: np.ndarray, gamma: float = 0.5) -> float`**

Computes the **epBIC** score for the estimated matrix `Xn`.

- **Inputs**:  
  - `Xn`: Estimated matrix (e.g., sparse precision estimate).  
  - `S`: Empirical covariance matrix.  
  - `gamma`: Tuning parameter for model complexity (default = 0.5).

- **Returns**:  
  A scalar epBIC score combining fit and model complexity.

- **Formula**:

\$\text{epBIC}(\lambda)=(2n)g(\hat{\Omega})+||\hat{\Omega}||_0\log n+4\gamma||\hat{\Omega}||_0\log p,~\gamma\in(0,1]$

### runner.py

This module provides a set of utilities for processing assay metadata and measurement matrices, computing correlation metrics (precision, partial, simple), validating input data, and saving results in structured formats.

### 🧠 Functions

**`parse_index_range(index_str: str) -> List[int]`**

Parses index strings like `"0,2,4-6"` into a list of integers:
**Example**: `"0,2,4-6"` → `[0, 2, 4, 5, 6]`


**`read_data(file_path: str) -> Tuple[pd.Index, np.ndarray]`**

Reads data from a `.xlsx`, `.xls`, or `.csv` file and returns the **header** and **data** separately.

* **Returns**:

  * `header`: Pandas Index object
  * `data`: NumPy array of shape `(n_samples, n_features)`


**`validate_numeric_2d_array(arr: np.ndarray) -> np.ndarray`**

Ensures that the input array is a **2D numeric NumPy array** with no `NaN` or non-numeric values.


**`convert_oid(oid: str) -> str`**

Converts OlinkID to assay name using the lookup dictionary from `OLINKprot_meta_data.txt`.


**`sign(number: float) -> str`**

Returns `'+'` or `'-'` depending on the sign of a number.


**`save_data(header: pd.Index, omega: np.ndarray, output_file: str)`**

Saves:

1. The `omega` matrix as a `.npy` file.
2. A transformed correlation table (`theta`, partial & simple correlations) as `.xlsx`, `.xls`, or `.csv`.

* The saved table includes:

  * Assay pairs (`V1`, `V2`)
  * Precision matrix values
  * Partial and simple correlations
  * Original assay names (`A`, `B`)
  * Absolute and sign of correlations


**`transform_data(header: pd.Index, data: np.ndarray) -> List[Tuple]`**

Builds a row-wise list of pairwise combinations with precision, partial, and simple correlations (excluding diagonals).


**`reconstruct_data(file_path: str) -> np.ndarray`**

Loads a previously saved `.npy` matrix file and returns it as a NumPy array.
