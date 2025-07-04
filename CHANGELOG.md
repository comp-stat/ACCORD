## [1.1.4] - 2025-06-02
### Modified
- Refactored to avoid sparseView(); further enhancement will follow.

## [1.1.3] - 2025-05-26
### Modified
- Save only the upper triangular part of the main output file.

## [1.1.2] - 2025-05-23
### Added 
- Add an option for ensuring sample label in the column-wise input data

## [1.1.1] - 2025-05-22
### Modified
- Switch the behavior of --row-wise and --column-wise options.
- Print the version of ACCORD for running.

## [1.1.0] - 2025-05-20
### Added
- Enable column-wise input data(p by n) via `--column-wise` option
- Add sparse output option(default) to save only zero-precision row via `--spasrse` option
- Change simple correlation of output using estimated precision matrix to pearson correlation using raw data
