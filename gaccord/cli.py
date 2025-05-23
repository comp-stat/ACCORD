import click
import pandas as pd
import numpy as np
from pathlib import Path
import traceback
from gaccord.runner import (
    read_data,
    validate_numeric_2d_array,
    save_data,
    parse_index_range,
    reconstruct_data,
)
from gaccord.gaccord import GraphicalAccord
from gaccord import __version__
import traceback


def parse_lam(lam_input):
    """
    Parse the lam1 or lam2 input, which can be:
    - A single float value
    - A space-separated list of float values
    - A range in the format `start:end:step`
    """
    if isinstance(lam_input, (float, int)):
        return np.array([lam_input])

    if isinstance(lam_input, str):
        if ":" in lam_input:  # Range format "start:end:step"
            parts = lam_input.split(":")
            if len(parts) != 3:
                raise ValueError("Range format must be 'start:end:step'")
            start, end, step = map(float, parts)
            return np.arange(start, end + step, step)  # Include end value
        else:  # List of values "0.1 0.2 0.3"
            return np.array([float(x) for x in lam_input.split()])

    raise ValueError("Invalid lam input format")


def validate_gamma(ctx, param, value):
    """Validate gamma to ensure it is in the range (0,1]."""
    gamma = float(value)
    if gamma <= 0 or gamma > 1:
        raise click.BadParameter("Gamma must be in the range (0,1].")
    return gamma


@click.command()
@click.version_option(__version__, prog_name="accord")
@click.option(
    "--input-file",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to the input file",
)
@click.option(
    "--column-wise/--row-wise",
    default=True,
    show_default=True,
    help="Orientation of the variables. --column-wise means the input data is n by p shaped.",
)
@click.option(
    "--output-file",
    "-o",
    type=click.Path(),
    required=True,
    help="Path to the output file",
)
@click.option(
    "--sparse/--dense", default=True, help="Save output in sparse format if enabled"
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
    type=str,
    default="0.0",
    show_default=True,
    help="Scalar value, space-separated list, or range (start:end:step)",
)
@click.option(
    "--gamma",
    type=float,
    default=0.2,
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
    "--include-sample-label",
    type=bool,
    show_default=True,
    help="""
        Whether the input data include label of the samples or not.
        For column-wise data, default value is True and False for row-wise data.
        """
)
@click.option(
    "--include-vars",
    type=str,
    default="",
    help="Comma-separated list of variable names to include",
)
@click.option(
    "--exclude-vars",
    type=str,
    default="",
    help="Comma-separated list of variable names to exclude",
)
@click.option(
    "--include-index",
    type=str,
    default="",
    help="Comma-separated list or range of variable indices to include (e.g., '0,2,4-6')",
)
@click.option(
    "--exclude-index",
    type=str,
    default="",
    help="Comma-separated list or range of variable indices to exclude (e.g., '1,3,5-7')",
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
    column_wise,
    output_file,
    sparse,
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
    include_sample_label,
    include_vars,
    exclude_vars,
    include_index,
    exclude_index,
    itr_logging_interval,
    warmup_file,
):
    try:
        if (
            sum(
                1
                for s in [include_vars, exclude_vars, include_index, exclude_index]
                if s != ""
            )
            >= 2
        ):
            raise ValueError(
                "Only one of the following options can be used: --include-vars, --exclude-vars, --include-index, --exclude-index"
            )

        lam1_values = parse_lam(lam1)
        lam2_values = parse_lam(lam2)

        # print version
        click.echo(f"[LOG] ACCORD version {__version__}")

        # 설정된 옵션 출력
        click.echo(f"[LOG] Processing with the following parameters:")
        click.echo(f"  Input File: {input_file}")
        click.echo(
            f"  Orientation of the variables: {'column-wise' if column_wise else 'row-wise'}"
        )
        click.echo(f"  Output File: {output_file}")
        click.echo(f"  Save Output File as: {'sparse' if sparse else 'dense'} format")
        click.echo(f"  Warm up File: {warmup_file}")
        click.echo(f"  L1 Regularization (λ1): {lam1_values}")
        click.echo(f"  The constant for epBIC (𝛾): {gamma}")
        click.echo(f"  L2 Regularization (λ2): {lam2_values}")
        click.echo(f"  Split Method: {split}")
        click.echo(f"  Stepsize Multiplier: {stepsize_multiplier}")
        click.echo(f"  Constant Stepsize: {constant_stepsize}")
        click.echo(f"  Backtracking: {'Enabled' if backtracking else 'Disabled'}")
        click.echo(f"  Convergence Threshold: {epstol}")
        click.echo(f"  Max Iterations: {maxitr}")
        click.echo(f"  Penalize Diagonal: {'Yes' if penalize_diag else 'No'}")
        click.echo(f"  Iteration Logging Interval: {itr_logging_interval}")

        (header, data) = read_data(input_file, column_wise)

        # Handle the label of samples
        if include_sample_label is None:
            include_sample_label = True if column_wise else False
        if include_sample_label:
            click.echo(f"  Exclude the label of samples")
            data = data[:, 1:]
            header = header[1:]

        if include_vars:
            include_vars_list = [
                variable.strip() for variable in include_vars.split(",")
            ]
            # 컬럼이 포함될지 여부
            data = data[:, np.isin(header, include_vars_list)]
            header = header[np.isin(header, include_vars_list)]
            click.echo(f"[LOG] Including variables: {include_vars_list}")

        if exclude_vars:
            exclude_vars_list = [
                variable.strip() for variable in exclude_vars.split(",")
            ]
            # 제외할 컬럼 처리
            data = data[:, ~np.isin(header, exclude_vars_list)]
            header = header[~np.isin(header, exclude_vars_list)]
            click.echo(f"[LOG] Excluding variables: {exclude_vars_list}")

        if include_index:
            include_index_list = parse_index_range(include_index)
            # 인덱스를 기반으로 포함
            data = data[:, include_index_list]
            header = header[include_index_list]
            click.echo(f"[LOG] Including variables by index: {include_index_list}")

        if exclude_index:
            exclude_index_list = parse_index_range(exclude_index)
            # 인덱스를 기반으로 제외
            data = np.delete(data, exclude_index_list, axis=1)
            header = [
                variable
                for i, variable in enumerate(header)
                if i not in exclude_index_list
            ]
            click.echo(f"[LOG] Excluding variables by index: {exclude_index_list}")

        header = pd.Index(header)
        data = validate_numeric_2d_array(data)
        click.echo(f"[LOG] Shape of input data is {data.shape[0]} x {data.shape[1]}")

        model_accord = GraphicalAccord(
            Omega_star=np.eye(len(header)),
            lam1_values=lam1_values,
            gamma=gamma,
            lam2_values=lam2_values,
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
        save_data(header, data, omega, output_file, sparse)

        # save epBIC tables
        rows = []
        for i, lam1 in enumerate(lam1_values):
            for j, lam2 in enumerate(lam2_values):
                idx = i * len(lam2_values) + j
                rows.append(
                    {
                        "lambda1": lam1,
                        "lambda2": lam2,
                        "epBIC": model_accord.epbic_values[idx],
                    }
                )
        df_epBIC = pd.DataFrame(rows)
        path = Path(output_file)
        epbic_path = str(path.with_name(path.stem + "_epBIC.csv"))
        df_epBIC.to_csv(epbic_path, index=False)
        print(f"[LOG] epBIC table saved to {epbic_path}")
    except Exception as e:
        traceback.print_exc()
        click.echo(f"Error: {e}")


if __name__ == "__main__":
    main()
