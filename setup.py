import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import subprocess
from setuptools import setup, Extension
from build_scripts import get_pybind_include, BuildExt

__version__ = "0.0.1"

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
    name="accord-reproducibility",
    version=__version__,
    author="Joshua Bang",
    author_email="joshuaybang@gmail.com",
    url="https://github.com/joshuaybang/accord-reproducibility",
    description="Reproducibility for the ACCORD simulations",
    long_description="",
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
    entry_points={
        "console_scripts": [
            "gaccord=gaccord.cli:main",
        ],
    },
)
