import click
import pandas as pd
import numpy as np
import traceback
from gaccord.runner import (
    read_data,
    validate_numeric_2d_array,
    save_data,
    parse_index_range,
    reconstruct_data,
)
from gaccord.gaccord import GraphicalAccord
import traceback


def parse_lam1(lam1_input):
    """
    Parse the lam1 input, which can be:
    - A single float value
    - A space-separated list of float values
    - A range in the format `start:end:step`
    """
    if isinstance(lam1_input, (float, int)):
        return np.array([lam1_input])

    if isinstance(lam1_input, str):
        if ":" in lam1_input:  # Range format "start:end:step"
            parts = lam1_input.split(":")
            if len(parts) != 3:
                raise ValueError("Range format must be 'start:end:step'")
            start, end, step = map(float, parts)
            return np.arange(start, end + step, step)  # Include end value
        else:  # List of values "0.1 0.2 0.3"
            return np.array([float(x) for x in lam1_input.split()])

    raise ValueError("Invalid lam1 input format")


def validate_gamma(ctx, param, value):
    """Validate gamma to ensure it is in the range (0,1]."""
    gamma = float(value)
    if gamma <= 0 or gamma > 1:
        raise click.BadParameter("Gamma must be in the range (0,1].")
    return gamma


@click.command()
@click.option(
    "--input-file",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to the input file",
)
@click.option(
    "--output-file",
    "-o",
    type=click.Path(),
    required=True,
    help="Path to the output file",
)
@click.option(
    "--lam1",
    type=str,
    default="0.1:1:0.1",
    show_default=True,
    help="Scalar value, space-separated list, or range (start:end:step)",
)
@click.option(
    "--lam2",
    type=float,
    default=0.0,
    show_default=True,
    help="The L2-regularization parameter",
)
@click.option(
    "--gamma",
    type=float,
    default=0.5,
    show_default=True,
    callback=validate_gamma,
    help="Gamma parameter in the range (0,1].",
)
@click.option(
    "--split",
    type=click.Choice(["fbs", "ista"], case_sensitive=False),
    default="fbs",
    show_default=True,
    help="The type of split",
)
@click.option(
    "--stepsize-multiplier",
    type=float,
    default=1.0,
    show_default=True,
    help="Multiplier for stepsize",
)
@click.option(
    "--constant-stepsize",
    type=float,
    default=0.5,
    show_default=True,
    help="Constant step size",
)
@click.option(
    "--backtracking/--no-backtracking",
    default=True,
    show_default=True,
    help="Whether to perform backtracking with lower bound",
)
@click.option(
    "--epstol",
    type=float,
    default=1e-5,
    show_default=True,
    help="Convergence threshold",
)
@click.option(
    "--maxitr",
    type=int,
    default=100,
    show_default=True,
    help="The maximum number of iterations",
)
@click.option(
    "--penalize-diag/--no-penalize-diag",
    default=True,
    show_default=True,
    help="Whether to penalize the diagonal elements",
)
@click.option(
    "--include-cols",
    type=str,
    default="",
    help="Comma-separated list of column names to include",
)
@click.option(
    "--exclude-cols",
    type=str,
    default="",
    help="Comma-separated list of column names to exclude",
)
@click.option(
    "--include-index",
    type=str,
    default="",
    help="Comma-separated list or range of column indices to include (e.g., '0,2,4-6')",
)
@click.option(
    "--exclude-index",
    type=str,
    default="",
    help="Comma-separated list or range of column indices to exclude (e.g., '1,3,5-7')",
)
@click.option(
    "--itr-logging-interval",
    type=int,
    default=100,
    show_default=True,
    help="The number of iterations until logging",
)
@click.option(
    "--warmup-file", type=click.Path(), default=None, help="Path to the warm up file"
)
def main(
    input_file,
    output_file,
    lam1,
    lam2,
    gamma,
    split,
    stepsize_multiplier,
    constant_stepsize,
    backtracking,
    epstol,
    maxitr,
    penalize_diag,
    include_cols,
    exclude_cols,
    include_index,
    exclude_index,
    itr_logging_interval,
    warmup_file,
):
    try:
        if (
            sum(
                1
                for s in [include_cols, exclude_cols, include_index, exclude_index]
                if s != ""
            )
            >= 2
        ):
            raise ValueError(
                "Only one of the following options can be used: --include-cols, --exclude-cols, --include-index, --exclude-index"
            )

        lam1_values = parse_lam1(lam1)

        # 설정된 옵션 출력
        click.echo(f"Processing with the following parameters:")
        click.echo(f"  Input File: {input_file}")
        click.echo(f"  Output File: {output_file}")
        click.echo(f"  Warm up File: {warmup_file}")
        click.echo(f"  L1 Regularization (λ1): {lam1_values}")
        click.echo(f"  The constant for epBIC (𝛾): {gamma}")
        click.echo(f"  L2 Regularization (λ2): {lam2}")
        click.echo(f"  Split Method: {split}")
        click.echo(f"  Stepsize Multiplier: {stepsize_multiplier}")
        click.echo(f"  Constant Stepsize: {constant_stepsize}")
        click.echo(f"  Backtracking: {'Enabled' if backtracking else 'Disabled'}")
        click.echo(f"  Convergence Threshold: {epstol}")
        click.echo(f"  Max Iterations: {maxitr}")
        click.echo(f"  Penalize Diagonal: {'Yes' if penalize_diag else 'No'}")
        click.echo(f"  Iteration Logging Interval: {itr_logging_interval}")

        (header, data) = read_data(input_file)

        if include_cols:
            include_cols_list = [col.strip() for col in include_cols.split(",")]
            # 컬럼이 포함될지 여부
            data = data[:, np.isin(header, include_cols_list)]
            header = header[np.isin(header, include_cols_list)]
            click.echo(f"Including columns: {include_cols_list}")

        if exclude_cols:
            exclude_cols_list = [col.strip() for col in exclude_cols.split(",")]
            # 제외할 컬럼 처리
            data = data[:, ~np.isin(header, exclude_cols_list)]
            header = header[~np.isin(header, exclude_cols_list)]
            click.echo(f"Excluding columns: {exclude_cols_list}")

        if include_index:
            include_index_list = parse_index_range(include_index)
            # 인덱스를 기반으로 포함
            data = data[:, include_index_list]
            header = header[include_index_list]
            click.echo(f"Including columns by index: {include_index_list}")

        if exclude_index:
            exclude_index_list = parse_index_range(exclude_index)
            # 인덱스를 기반으로 제외
            data = np.delete(data, exclude_index_list, axis=1)
            header = [
                col for i, col in enumerate(header) if i not in exclude_index_list
            ]
            click.echo(f"Excluding columns by index: {exclude_index_list}")

        header = pd.Index(header)
        data = validate_numeric_2d_array(data)
        click.echo(f"Shape of input data is {data.shape[0]} x {data.shape[1]}")

        model_accord = GraphicalAccord(
            Omega_star=np.eye(len(header)),
            lam1_values=lam1_values,
            gamma=gamma,
            lam2=lam2,
            split=split,
            stepsize_multiplier=stepsize_multiplier,
            constant_stepsize=constant_stepsize,
            backtracking=backtracking,
            epstol=epstol,
            maxitr=maxitr,
            penalize_diag=penalize_diag,
            logging_interval=itr_logging_interval,
        )

        if warmup_file is not None:
            model_accord.fit(data, initial=reconstruct_data(warmup_file))
        else:
            model_accord.fit(data)

        omega = model_accord.omega_.toarray()
        save_data(header, omega, output_file)
    except Exception as e:
        traceback.print_exc()
        click.echo(f"Error: {e}")


if __name__ == "__main__":
    main()
