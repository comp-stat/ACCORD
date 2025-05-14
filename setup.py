import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import subprocess
from setuptools import setup, Extension, find_packages
from build_scripts import get_pybind_include, BuildExt

__version__ = "1.0.0-alpha"

includes = [get_pybind_include(), get_pybind_include(user=True), "eigen-3.4.0"]

ext_modules = [
    Extension(
        "_gaccord",
        ["src/main.cpp", "src/core.cpp"],
        include_dirs=includes,
        extra_compile_args=["-std=libc++"],
        language="c++",
    ),
]

setup(
    name="accord",
    version=__version__,
    url="https://github.com/comp-stat/ACCORD",
    description="Python-based CLI application for ACCORD",
    ext_modules=ext_modules,
    install_requires=[
        "pybind11>=2.2",  # needs to be updated
        "click",
        "numpy",
        "pandas",
        "scipy",
        "openpyxl",
        "psutil",
    ],
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    packages=find_packages(exclude=['data*', 'tests*']),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "accord=gaccord.cli:main",
        ],
    },
)
