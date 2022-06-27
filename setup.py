#!/usr/bin/env python
import os
from setuptools import setup

ROOT = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
    README = f.read()
with open(os.path.join("uaml", "version.txt")) as f:
    VERSION = f.read().strip()

setup(
    name="uaml",
    version=VERSION,
    license="MIT license",
    description="Uncertainty-aware classification.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Thomas Mortier",
    author_email="thomas.mortier92@gmail.com",
    url="https://github.com/tfmortie/uaml",
    packages=["uaml"],
    install_requires=[
        "numpy",
        "scikit-learn>=0.23.0",
        "setuptools",
    ],
    include_package_data=True,
)
