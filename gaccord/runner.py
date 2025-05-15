import pandas as pd
import numpy as np
from gaccord.correlation import compute_partial_correlation, compute_simple_correlation
import csv
import os
from pathlib import Path
from importlib.resources import files

assay_by_oid = {}
with files("gaccord").joinpath("OLINKprot_meta_data.txt").open(
    "r", encoding="utf-8"
) as file:
    reader = csv.DictReader(file, delimiter="\t")
    for row in reader:
        olink_id = row["OlinkID"]
        assay = row["Assay"]
        assay_by_oid[olink_id] = assay


def parse_index_range(index_str):
    """문자열로 입력된 인덱스 범위를 처리하는 함수. 예: '0,2,4-6'"""
    indices = []
    for part in index_str.split(","):
        if "-" in part:
            start, end = part.split("-")
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return indices


def read_data(file_path):
    """
    Reads an xlsx, xls, or csv file and returns header and data separately.

    :param file_path: Path to the xlsx, xls, or csv file
    :return: (header, data) -> header is a list, data is a list of lists
    """
    # 파일 확장자 확인
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path, engine="openpyxl")  # XLSX 처리
    elif file_path.endswith(".xls"):
        df = pd.read_excel(file_path, engine="xlrd")  # XLS 처리
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)  # CSV 처리
    else:
        raise ValueError("Unsupported file format. Please use .xlsx, .xls, or .csv")

    # 헤더와 데이터 분리
    header = df.columns  # 헤더 리스트
    data = df.values  # 데이터 리스트

    return header, data


def validate_numeric_2d_array(arr):
    """
    Checks if a 2D ndarray contains only numeric values and no None or NaN.

    :param arr: 2D numpy ndarray to check
    :raises ValueError: if any value is not numeric or is None/NaN
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")

    if arr.ndim != 2:
        raise ValueError("Input must be a 2D ndarray.")

    # Check for None/NaN and non-numeric values
    for i in range(arr.shape[0]):  # Iterate over rows
        for j in range(arr.shape[1]):  # Iterate over columns
            value = arr[i, j]
            if pd.isna(value):  # Check for None or NaN
                raise ValueError(
                    f"None (NaN) value found at position ({i}, {j}): {value}"
                )
            if not np.issubdtype(type(value), np.number):  # Check if value is numeric
                raise ValueError(
                    f"Non-numeric value found at position ({i}, {j}): {value}"
                )

    return arr.astype(np.float64)


def convert_oid(oid):
    return assay_by_oid.get(oid, oid)


def sign(number):
    return "+" if number >= 0 else "-"


def save_data(header, omega, output_file):
    """
    Saves data into an xlsx, xls, or csv file.

    :param header: List of column names
    :param omega: List of lists (table data)
    :param output_file: Output file path with extension
    """

    npy_file = str(Path(output_file).with_suffix(f".npy"))
    np.save(npy_file, omega)
    print(f"[LOG] Omega saved to {npy_file}")

    # Omega -> Theta
    D = np.diag(np.diag(omega))
    theta = D @ omega
    theta = 0.5 * (theta + theta.T)

    # DataFrame 생성
    df = pd.DataFrame(
        transform_data(header, theta),
        columns=[
            "V1",
            "V2",
            "Precision.value",
            "Partial.Corr",
            "Simple.Corr",
            "A",
            "B",
            "AbsPartialCorr",
            "SignPartialCorr",
            "AbsSimpleCorr",
            "SignSimpleCorr",
        ],
    )

    # 확장자 확인 후 저장
    if output_file.endswith(".xlsx"):
        df.to_excel(output_file, index=False, engine="openpyxl")
    elif output_file.endswith(".xls"):
        df.to_excel(output_file, index=False, engine="xlwt")
    elif output_file.endswith(".csv"):
        df.to_csv(output_file, index=False)
    else:
        raise ValueError("Unsupported file format. Please use .xlsx, .xls, or .csv")

    print(f"[LOG] Data saved to {output_file}")


def transform_data(header, data):
    """
    Transform a 2D ndarray into a list of tuples where each tuple contains (header[i], header[j], data[i][j]),
    but only for the upper triangle (excluding the diagonal).

    :param header: pandas Index object (e.g., a row or column index)
    :param data: 2D numpy ndarray with values corresponding to the header combination
    :return: List of tuples in the form (header[i], header[j], data[i][j]), only for upper triangle (i < j)
    """
    if not isinstance(header, pd.Index):
        raise TypeError("header must be a pandas Index")

    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy ndarray")

    P = compute_partial_correlation(data)
    D = compute_simple_correlation(data)

    result = []
    for i in range(header.shape[0]):  # Iterate over rows of header
        for j in range(header.shape[0]):  # Iterate over columns of header
            if i == j:
                continue
            result.append(
                (
                    header[i],
                    header[j],
                    data[i][j],
                    P[i][j],
                    D[i][j],
                    convert_oid(header[i]),
                    convert_oid(header[j]),
                    abs(P[i][j]),
                    sign(P[i][j]),
                    abs(D[i][j]),
                    sign(D[i][j]),
                )
            )

    return result


def reconstruct_data(file_path):
    return np.load(file_path)
